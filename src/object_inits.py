from prefect import task, get_run_logger
from src.data_source.data_source_class import DataSource
from src.reporting.reporter_class import Reporter
from src.quality_assurance_control.quality_ac_class import QualityAC
from src.reward.reward_class import Reward
from src.session_management.session_manager import SessionManager
from src.experiment_tracker.experiment_tracker_class import ExperimentTracker

# Session manager acts to parse tables that contain past and new sessions for analysis.
@task(name="init_session_manager")
def init_session_manager(config):
    logger = get_run_logger()
    logger.info(f"Initializing SessionManager with config: {config}")
    return SessionManager(config)

# Experiment tracker acts to track the state of the experiment, including the participants, past sessions, and parameters.
@task(name="init_experiment_tracker")
def init_experiment_tracker(config):
    logger = get_run_logger()
    logger.info(f"Initializing ExperimentTracker with config: {config}")
    return ExperimentTracker(config)

# Data source acts to get data from the data source.
@task(name="init_data_source") 
def init_data_source(config):
    logger = get_run_logger()
    logger.info(f"Initializing DataSource with config: {config}")
    return DataSource(config)

# Reporter acts to save and visualize data (e.g. plots, tables) to databases, W&B, local files, etc.
@task(name="init_reporter")
def init_reporter(config):
    logger = get_run_logger()
    logger.info(f"Initializing Reporter with config: {config}")
    return Reporter(config)

# Quality Assurance/Control acts to check the quality of the session data.
@task(name="init_quality_ac")
def init_quality_ac(config):
    logger = get_run_logger()
    logger.info(f"Initializing QualityAC with config: {config}")
    return QualityAC(config)

# Reward acts to calculate the reward function from the session data.
@task(name="init_reward")
def init_reward(config):
    logger = get_run_logger()
    logger.info(f"Initializing Reward with config: {config}")
    return Reward(config)