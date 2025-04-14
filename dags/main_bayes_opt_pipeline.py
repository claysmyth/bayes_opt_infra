from prefect import flow, tags, task, get_run_logger
import polars as pl
import os
from omegaconf import DictConfig, OmegaConf
import hydra
from src.object_inits import *


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
    session_manager = init_session_manager(config["session_manager_config"])
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
    reporter = init_reporter(config["reporting_config"])

    # Get sessions which have not been reported yet.
    sessions_info = session_manager.get_new_sessions(run_tasks=True)

    participant_column = config["participant_column"]

    # Cycle sessions_info and get sessions for each participant
    for participant in sessions_info.partition_by(participant_column):
        with tags(participant[participant_column]):  # Add device as prefect tag
            experiment_id = participant[experiment_tracker.experiment_id_column]

            # Update experimental context (e.g. participant id, experiment id, etc...)
            # TODO: Check if experiment is initialized. If not, attempt to initialize it, or throw error that experiment is not initialized. Point user to scr/experiment_tracker/initialize_experiment.py
            experiment_tracker.update_context(participant, experiment_id)

            # Create data source object
            session_data = data_source.get_data(
                participant
            )  # Probably pull data from processed parquet files, where sessions where aggregated.

            if quality_ac is not None:
                # QA Check (e.g. check if session settings are as expected)
                # QA is to prevent bad sessions from being included in optimization.
                bad_data = quality_ac.quality_assurance_check(
                    session_data
                )  # Will return None if no QA checks are configured. True if any QA checks fail.
                if bad_data:
                    session_manager.mark_session_as_bad(participant)
                    continue

                # QC Check (check if session data is as expected (e.g. less than 50% null))
                # QC is to hold sessions with insufficient data, until enough data is collected to pass QC,
                # so that an appropriate evaluate observation can be calculated.
                insufficient_data = quality_ac.quality_control_check(
                    session_data
                )  # Will return None if no QC checks are configured. True if any QC checks fail.
                if insufficient_data:
                    session_manager.mark_session_as_insufficient(participant)
                    continue

                # Evaluation Function (calculate evaluation function from data)
                result = evaluation.evaluate(session_data)

                # Visualize the data and log to relevant dashboards (e.g. WandB, prefect, and local)
                # TODO: Define analyses... or ingore if analyses should belong to sleep_aDBS_infra
                path = reporter.run(session_data, analyses, participant)

                # Update Bayesian Optimizer with new settings
                experiment_tracker.update_optimizer(result)
                parameters_to_try = experiment_tracker.get_parameters_to_try()

                # Generate final format of new parameters and ship to relevant destination
                # (e.g. generate new RC+S adaptive config file from parameters, and send to device)
                experiment_tracker.save_parameters(parameters_to_try)
                shipped_file = parameter_shipment.ship_parameters_to_destination(
                    parameters_to_try
                )

                # Save and visualize new settings and Bayes Opt updates. Save file sent to device
                # ? Save through experiment tracker or reporter?
                # TODO: How to save everything??

                # Add session to reported sessions csv
                session_manager.update_reported_sessions(session)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config_main",  # Would need to swap 'config_main' for a different bayes opt project (like Muse)
)
def hydra_main_pipeline(cfg: DictConfig):
    bayes_opt_main_pipeline(cfg)


if __name__ == "__main__":
    hydra_main_pipeline()
