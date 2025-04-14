from prefect import task
from typing import Dict, Any
from pathlib import Path


@task(name="ship_rcs_parameters")
def ship_rcs_parameters(parameters: Dict[str, Any], config: Dict[str, Any]) -> str:
    """
    Ship parameters to RC+S device by creating adaptive config file.

    Args:
        parameters: Dictionary of parameters from Bayesian optimization
        config: Configuration dictionary with paths and settings

    Returns:
        Path to the created config file
    """
    # Create RC+S config from parameters
    device_config = create_rcs_config(parameters)

    # Save config file
    output_path = (
        Path(config["paths"]["base_dir"])
        / f"adaptive_config_{parameters['trial_index']}.json"
    )
    device_config.save(output_path)

    # In real implementation, would send to device here
    # send_to_device(output_path)

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
