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
