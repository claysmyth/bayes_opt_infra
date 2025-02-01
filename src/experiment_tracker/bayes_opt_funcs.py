from typing import Dict, Any
from src.experiment_tracker.experiment_tracker_class import ExperimentTracker

def complete_and_get_trial(experiment_tracker: ExperimentTracker, evaluation_result: Dict[str, float]) -> Dict[str, Any]:
    """
    Complete current trial with evaluation results and get next trial parameters.
    
    Args:
        experiment_tracker: ExperimentTracker instance
        evaluation_result: Dictionary with objective name(s) and value(s)
        
    Returns:
        Dictionary with next trial parameters
    """
    # Complete current trial
    trial_index = experiment_tracker.ax_client.generation_strategy.current_trial_index
    experiment_tracker.ax_client.complete_trial(
        trial_index=trial_index,
        raw_data=evaluation_result
    )
    
    # Get next trial
    parameters, trial_index = experiment_tracker.ax_client.get_next_trial()
    
    return parameters
