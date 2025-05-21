import os
import polars as pl
from typing import Dict, Any, Optional
from pathlib import Path
import wandb
import json
import tempfile
import shutil
# from src.experiment_tracker.experiment_tracker_class import ExperimentTracker

def init_local_reporting_rcs(
    participant_sessions: pl.DataFrame,
    experiment_tracker: 'ExperimentTracker',  # Use string literal to avoid circular import
    reporting_path_base: str = None
) :
    """
    Initialize local reporting for RCS data with trial-specific paths.
    
    Parameters
    ----------
    participant_sessions : pl.DataFrame
        DataFrame containing participant session information
    experiment_tracker : ExperimentTracker
        Experiment tracker object to get current trial index
    reporting_path_base : str
        Base path for reporting
        
    Returns
    -------
    Optional[str]
        Path to trial-specific reporting directory if created, None otherwise
    """
    try:
        # Get unique participant from RCS# column
        participant = participant_sessions.get_column("RCS#").unique().to_list()[0]
        
        # Get current trial index from experiment tracker
        trial_index = experiment_tracker.get_trial_index()
        
        # Format reporting path with participant and trial
        reporting_path = Path(reporting_path_base.format(participant=participant))
        trial_path = reporting_path / str(trial_index)
        
        # Create trial directory if it doesn't exist
        if not trial_path.exists():
            trial_path.mkdir(parents=True, exist_ok=True)
            
        return str(trial_path)
        
    except Exception as e:
        print(f"Error initializing local reporting: {e}")
        return None
    
    
def log_rcs_shipment(
    shipped_files: Dict[str, Any],
    experiment_tracker: 'ExperimentTracker',
    local_path: Optional[str] = None
) -> None:
    """
    Log RCS shipped files for each trial to corresponding local trial directory for saving.

    Parameters
    ----------
    shipped_files : Dict[str, Any]
        Dictionary containing shipped files information
    """
    files_to_save_to_local = []
    adaptive_config_paths = shipped_files['ship_rcs_adaptive_configs_to_device']
    for key, path in adaptive_config_paths.items():
        files_to_save_to_local.append(path)

    files_to_save_to_local.append(shipped_files['update_current_target_amp_cache']['cache_path'])

    # Copy files to local path
    for file_path in files_to_save_to_local:
        shutil.copy(file_path, os.path.join(local_path, os.path.basename(file_path)))

    # Save experiment tracker to local path
    experiment_tracker.save_experiment_to_json_file(os.path.join(local_path, 'experiment_snapshot.json'))
    

def init_wandb_rcs(
    participant_sessions: pl.DataFrame,
    experiment_tracker: 'ExperimentTracker',  # Use string literal to avoid circular import
    wandb_config: Dict[str, Any],
    local_path: Optional[str] = None
):
    """
    Initialize Weights & Biases for RCS data with trial-specific information.
    
    Parameters
    ----------
    participant_sessions : pl.DataFrame
        DataFrame containing participant session information
    experiment_tracker : ExperimentTracker
        Experiment tracker object to get current trial information
    wandb_config : Dict[str, Any]
        Configuration dictionary for Weights & Biases
    local_path : Optional[str]
        Path to local reporting directory
        
    Returns
    -------
    Optional[wandb.Run]
        Initialized Weights & Biases run if successful, None otherwise
    """
    try:
        # Get current trial information
        current_trial = experiment_tracker.get_current_trial()
        trial_index = current_trial.get("index", 0)
        
        # Add session and device relevant info to wandb_config
        wandb_config["tags"] = participant_sessions.get_column("Session#").unique().to_list()
        wandb_config["job_type"] = participant_sessions.get_column("RCS#").unique().to_list()[0]
        wandb_config["group"] = participant_sessions.get_column("SessionType(s)").unique().to_list()[0]
        
        # Add trial information to config
        wandb_config["trial_index"] = trial_index
        wandb_config["trial_parameters"] = current_trial.get("parameters", {})
        wandb_config["local_results_path"] = local_path
        
        # Initialize wandb
        run = wandb.init(
            project=wandb_config["project"],
            entity=wandb_config["entity"],
            job_type=wandb_config["job_type"],
            group=wandb_config["group"],
            tags=wandb_config["tags"],
        )
        
        # Log wandb_config as an artifact instead of config
        config_artifact = wandb.Artifact(
            name=f"trial_config_{trial_index}", 
            type="config"
        )
        
        # Create a temporary file to store the config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(wandb_config, f, indent=2)
            config_path = f.name
        
        # Add the config file to the artifact
        config_artifact.add_file(config_path)
        
        # Log the artifact
        run.log_artifact(config_artifact)
        
        # Clean up the temporary file
        os.remove(config_path)
        
        return run
        
    except Exception as e:
        print(f"Error initializing Weights & Biases: {e}")
        return None