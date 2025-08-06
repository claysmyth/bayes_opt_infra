from prefect import flow, tags
from prefect.settings import PREFECT_API_URL
from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path
import sys
from dotenv import load_dotenv
import os
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.object_inits import *

# Load environment variables from .env file
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Get and set the Prefect API URL
api_url = os.getenv('PREFECT_API_URL')
if api_url:
    os.environ["PREFECT_API_URL"] = api_url  # Set as environment variable instead
else:
    raise ValueError("PREFECT_API_URL not found in .env file")

@flow(log_prints=True)
def bayes_opt_main_pipeline(cfg: DictConfig):

    # Convert DictConfig to a dictionary
    config = OmegaConf.to_container(cfg, resolve=True)
    global_config = config["global_config"]

    # Initialize objects and log configs (all except bayes opt, which is initialized in the loop)
    # Two options for experiment tracker: A class that updates context based upon experiment id, or simply a config that calls functions for managing the experiment.
    # Manages Bayes opt experiments
    experiment_tracker = init_experiment_tracker(config["experiment_tracker_config"])

    # Manages sessions that have been reported, and need to be reported. Interacts with Project CSV
    session_manager = init_session_manager(config["session_management_config"])

    # Manages how to collect data from the data source
    data_source = init_data_source(config["data_source_config"])

    # Quality Assurance and Quality Control are optional, so only initialize if they are defined in the global config
    if config["quality_ac_config"] is not None:
        quality_ac = init_quality_ac(config["quality_ac_config"])
    else:
        quality_ac = None

    # Manages how to evaluate the data for bayes opt. I.e. calculate cost or reward from data.
    evaluation = init_evaluation(config["evaluation_config"])

    # Manages how to ship parameters to the device
    parameter_shipment = init_parameter_shipment(config["parameter_shipment_config"])

    # Manages visualizations and reports and corresponding filepaths
    reporter = init_reporter(config["reporter_config"])

    # Get sessions which have not been reported yet.
    sessions_info = session_manager.get_new_sessions(run_tasks=False)

    participant_column = config["participant_column"]

    # Cycle sessions_info and get sessions for each participant
    for participant_sessions in sessions_info.partition_by(participant_column):
        participant = participant_sessions.get_column(participant_column).unique().item()

        print(f"Processing participant {participant}...")
        with tags(participant):  # Add device as prefect tag

            # Update experimental context (e.g. participant id, experiment id, etc...)
            try:
                experiment_tracker.update_ax_client(participant)
            except Exception as e:
                print(f"Error updating experiment context: {e} for participant {participant}")
                print(f"Skipping participant {participant}...")
                continue

            try:
                # Create data source object
                session_data = data_source.get_data(
                    participant_sessions
                )
            except Exception as e:
                print(f"Error getting data: {e} for participant {participant}")
                print(f"Skipping participant {participant}...")
                continue

            if quality_ac is not None:
                # QA Check (e.g. check if session settings are as expected)
                # QA is to prevent bad sessions from being included in optimization.
                bad_data = quality_ac.quality_assurance_check(
                    session_data,
                    participant_sessions
                )  # Will return None if no QA checks are configured. True if any QA checks fail.
                if bad_data:
                    print(f"Failed QA checks for participant {participant}.")
                    session_manager.update_reported_sessions(participant_sessions, flags=["QA_FAILED"])
                    continue
                else:
                    print(f"Passed QA checks for participant {participant}.")

                # QC Check (check if session data is as expected (e.g. less than 50% null))
                # QC is to hold sessions with insufficient data, until enough data is collected to pass QC,
                # so that an appropriate evaluate observation can be calculated.
                insufficient_data = quality_ac.quality_control_check(
                    session_data,
                    participant_sessions
                )  # Will return None if no QC checks are configured. True if any QC checks fail.
                if insufficient_data:
                    print(f"Failed QC checks for participant {participant}.")
                    session_manager.update_reported_sessions(participant_sessions, flags=["QC_FAILED"])
                    continue
                else:
                    print(f"Passed QC checks for participant {participant}.")

            # Evaluation Function (calculate evaluation function from data)
            result = evaluation.evaluate(session_data, participant)

            # Update reporter with current participant sessions and experiment tracker.
            # Needs to be called before experiment_tracker.update_experiment(), so that currently running trial can be captured.
            reporter.update_for_current_participant(participant_sessions, experiment_tracker)

            # Update Bayesian Optimizer with results and get next parameters. Note update experiment does not call next trial.
            experiment_tracker.update_experiment(result)

            # Saving plots, tables, current parameter caches, etc...
            reporter.report_results(session_data, result, participant_sessions, experiment_tracker)

            next_trial = experiment_tracker.get_next_trial()

            # Ship new parameters to destination
            shipped_files = parameter_shipment.ship_parameters(
                parameters=next_trial["parameters"],
                participant_info=participant_sessions
            )

            reporter.reinit_local_reporting(participant_sessions, experiment_tracker)
            reporter.log_shipment(shipped_files, experiment_tracker)

            # Save session to reported sessions csv
            session_manager.update_reported_sessions(participant_sessions)

            # Save updated experiment back to the json file
            experiment_tracker.save_experiment_to_json()


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config_main",  # Would need to swap 'config_main' for a different bayes opt project (like Muse)
)
def hydra_main_pipeline(cfg: DictConfig):
    bayes_opt_main_pipeline(cfg)


if __name__ == "__main__":
    hydra_main_pipeline()
