from prefect import task
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
import shutil


def _load_template(template_path: Path) -> Dict[str, Any]:
    """Load JSON template file."""
    with open(template_path, 'r') as f:
        return json.load(f)


def _map_parameters_to_template(
    template: Dict[str, Any],
    parameters: Dict[str, Any],
    parameter_mapping: Dict[str, str]
) -> Dict[str, Any]:
    """
    Map parameters to template using dot notation mapping.
    
    Args:
        template: Base template dictionary
        parameters: Parameters from Ax optimization
        parameter_mapping: Dictionary mapping parameter names to template fields
            e.g. {'amplitude': 'Adaptive.Program0.State1AmpInMilliamps'}
    """
    updated_template = template.copy()
    
    for param_name, param_value in parameters.items():
        if param_name in parameter_mapping:
            # Get the template field path
            field_path = parameter_mapping[param_name].split('.')
            
            # Navigate to the correct location in the template
            current = updated_template
            for key in field_path[:-1]:
                current = current[key]
            
            # Update the value
            current[field_path[-1]] = param_value
    
    return updated_template


@task(name="ship_parameters")
def ship_parameters(
    experiment_tracker,
    parameters: Dict[str, Any],
    config: Dict[str, Any]
) -> Union[Dict[str, str], str]:
    """
    Ship parameters based on experiment tracker's stimulation mode.
    
    Args:
        experiment_tracker: ExperimentTracker instance
        parameters: Dictionary of parameters from Bayesian optimization
        config: Configuration dictionary with paths and settings
        
    Returns:
        Union[Dict[str, str], str]: Path(s) to created config file(s)
    """
    stimulation_mode = experiment_tracker.get_stimulation_mode()
    participant_id = experiment_tracker.current_participant_id
    
    if stimulation_mode == "bilateral":
        return _ship_bilateral_parameters(
            parameters=parameters,
            config=config,
            participant_id=participant_id
        )
    else:
        # For unilateral, get the side from the experiment config
        side = experiment_tracker.config["experiment"].get("stimulation_side", "L")
        return _ship_unilateral_parameters(
            parameters=parameters,
            config=config,
            participant_id=participant_id,
            side=side
        )


def _ship_bilateral_parameters(
    parameters: Dict[str, Any],
    config: Dict[str, Any],
    participant_id: str
) -> Dict[str, str]:
    """Internal function to ship bilateral parameters."""
    # Get template paths
    template_dir = Path(config["paths"]["template_dir"])
    left_template_path = template_dir / participant_id / "adaptive_config_L.json"
    right_template_path = template_dir / participant_id / "adaptive_config_R.json"
    
    # Load templates
    left_template = _load_template(left_template_path)
    right_template = _load_template(right_template_path)
    
    # Get parameter mappings
    left_mapping = config["parameter_mappings"]["left"]
    right_mapping = config["parameter_mappings"]["right"]
    
    # Update templates
    updated_left = _map_parameters_to_template(left_template, parameters, left_mapping)
    updated_right = _map_parameters_to_template(right_template, parameters, right_mapping)
    
    # Create output directory
    output_dir = Path(config["paths"]["output_dir"]) / participant_id / f"trial_{parameters['trial_index']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save updated configs
    left_output = output_dir / "adaptive_config_L.json"
    right_output = output_dir / "adaptive_config_R.json"
    
    with open(left_output, 'w') as f:
        json.dump(updated_left, f, indent=4)
    with open(right_output, 'w') as f:
        json.dump(updated_right, f, indent=4)
    
    return {
        "Left": str(left_output),
        "Right": str(right_output)
    }


def _ship_unilateral_parameters(
    parameters: Dict[str, Any],
    config: Dict[str, Any],
    participant_id: str,
    side: str
) -> str:
    """Internal function to ship unilateral parameters."""
    # Get template path
    template_dir = Path(config["paths"]["template_dir"])
    template_path = template_dir / participant_id / f"adaptive_config_{side}.json"
    
    # Load template
    template = _load_template(template_path)
    
    # Get parameter mapping
    mapping = config["parameter_mappings"][side.lower()]
    
    # Update template
    updated_template = _map_parameters_to_template(template, parameters, mapping)
    
    # Create output directory
    output_dir = Path(config["paths"]["output_dir"]) / participant_id / f"trial_{parameters['trial_index']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save updated config
    output_path = output_dir / f"adaptive_config_{side}.json"
    with open(output_path, 'w') as f:
        json.dump(updated_template, f, indent=4)
    
    return str(output_path)


@task(name="ship_test_parameters")
def ship_test_parameters(parameters: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Example shipment function for testing."""
    return f"Parameters shipped: {parameters}"


def update_json_config(filepath: str, update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update JSON config file with new parameter values.

    Args:
        filepath: Path to JSON config file
        update_dict: Dictionary of key-value pairs to update in the JSON. Keys can be nested using dot notation (e.g. 'field0.field1.field2')

    Returns:
        Updated JSON content as dictionary
    """
    import json

    # Read existing JSON file
    with open(filepath, "r") as f:
        config = json.load(f)

    # Update values
    for key_path, value in update_dict.items():
        # Split nested key path
        keys = key_path.split(".")

        # Navigate to the nested location
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the value at the final key
        current[keys[-1]] = value

    # Write back to file
    with open(filepath, "w") as f:
        json.dump(config, f, indent=4)

    return config
