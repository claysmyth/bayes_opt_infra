from typing import Dict, Any

def default_context_update(
    experiment_tracker,
    participant: Dict[str, Any],
    experiment_id: str
) -> None:
    """
    Default implementation of context update.
    Updates participant context and loads appropriate Ax client.
    """
    experiment_tracker._default_update_context(participant, experiment_id)

def bilateral_context_update(
    experiment_tracker,
    participant: Dict[str, Any],
    experiment_id: str
) -> None:
    """
    Bilateral-specific context update.
    Handles both left and right devices for a participant.
    """
    participant_id = participant[experiment_tracker.experiment_id_column].split('_')[0]  # e.g., 'RCS09' from 'RCS09_L'
    
    # If we're already working with this participant, no need to reload
    if participant_id == experiment_tracker.current_participant_id:
        return
        
    # Update current context
    experiment_tracker.current_participant_id = participant_id
    experiment_tracker.current_experiment_id = experiment_id
    
    # Load participant-specific Ax client
    experiment_tracker.ax_client = experiment_tracker._load_ax_client(participant_id)
