from typing import Dict, Any
from ax.service.ax_client import AxClient
import json

def get_rcs_sleep_aDBS_ax_client(participant: str, experiment: str, base_dir_template: str) -> None:
    """
    Get the Ax client for the RCS Sleep aDBS experiment.
    """
    filepath = base_dir_template.format(participant=participant, experiment=experiment)
    return (
        AxClient.load_from_json_file(filepath=filepath), 
        filepath
    )


def get_rcs_sleep_aDBS_ax_client_first_trial_parameters(participant: str, base_init_params_filepath: str) -> None:
    """
    Get the Ax client for the RCS Sleep aDBS experiment.
    """
    with open(base_init_params_filepath.format(participant=participant), "r") as f:
        init_params = json.load(f)
    
    return init_params