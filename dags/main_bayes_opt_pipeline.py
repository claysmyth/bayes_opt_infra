from prefect import flow, task, get_run_logger
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
    reporter = init_reporter(global_config["REPORTING_CONFIG"])
    experiment_tracker = init_experiment_tracker(global_config["EXPERIMENT_TRACKER_CONFIG"], reporter)
    session_manager = init_session_manager(global_config["SESSION_MANAGER_CONFIG"])
    data_source = init_data_source(global_config["DATA_SOURCE_CONFIG"])
    # Quality Assurance and Quality Control are optional, so only initialize if they are defined in the global config
    if global_config["QUALITY_AC_CONFIG"] is not None:
        quality_ac = init_quality_ac(global_config["QUALITY_AC_CONFIG"])
    else:
        quality_ac = None
    reward = init_reward(global_config["REWARD_CONFIG"])

    # Get sessions which have not been reported yet.
    sessions_info = session_manager.get_new_sessions(run_tasks=True)

    for session in sessions_info:
        experiment_id = session[global_config["EXPERIMENT_TRACKER_CONFIG"]["experiment_id_column"]]    

        # Update experimental context (e.g. participant id, experiment id, etc...)
        experiment_tracker.update_context(session, experiment_id)

        # Create data source object
        session_data = data_source.get_data(session) # Probably pull data from processed parquet files, where sessions where aggregated.

        if quality_ac is not None:
            # QA Check (e.g. check if session settings are as expected)
            # QA is to prevent bad sessions from being included in optimization.
            bad_data = quality_ac.quality_assurance_check(session_data)
            if bad_data:
                session_manager.mark_session_as_bad(session)
                continue

            # QC Check (check if session data is as expected (e.g. less than 50% null))
            # QC is to hold sessions with insufficient data, until enough data is collected to pass QC,
            # so that an appropriate reward observation can be calculated.
            insufficient_data = quality_ac.quality_control_check(session_data)
            if insufficient_data:
                session_manager.mark_session_as_insufficient_data(session)
                continue

        # Reward Function (calculate reward function from data)
        reward.reward_function(session_data)

        # Save to database and generate visualizations
        # ? Save through experiment tracker or reporter?

        # Bayesian Optimizer (update Bayesian Optimizer with reward function)
        # Call reinit to reformat the Bayesian Optimizer for the next session.
        bayes_opt = init_bayes_opt(global_config["BAYES_OPT_CONFIG"], reward, reporter, experiment_tracker)

        # Update Bayesian Optimizer with new settings
        bayes_opt.update_optimizer(session_data)
        parameters_to_try = bayes_opt.get_parameters_to_try()

        # Save and visualize new settings and Bayes Opt updates
        # ? Save through experiment tracker or reporter?

        # Generate final format of new parameters and ship to relevant destination
        # (e.g. generate new RC+S adaptive config file from parameters, and send to device)
        experiment_tracker.save_parameters(parameters_to_try)
        experiment_tracker.ship_parameters_to_destination(parameters_to_try)
    
        # Add session to reported sessions csv
        session_manager.update_reported_sessions(session)


@hydra.main(
    version_base=None, config_path="../configs", config_name="config_main"
)
def hydra_main_pipeline(cfg: DictConfig):
    bayes_opt_main_pipeline(cfg)


if __name__ == "__main__":
    hydra_main_pipeline()
