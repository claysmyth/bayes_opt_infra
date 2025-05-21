from prefect import task
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
import bidict
import polars as pl

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


def ship_rcs_adaptive_configs_to_device(parameters: Dict[str, Any], participant_info: pl.DataFrame, **config) -> Dict[str, Any]:
    """
    Ship RCS adaptive configs to the device by updating templates with new parameters.
    
    Parameters
    ----------
    parameters : Dict[str, Any]
        Dictionary containing the parameters to ship
    participant_info : pl.DataFrame
        DataFrame containing participant information
    config : Dict[str, Any]
        Configuration containing:
        - shipment_destination_path: str (path template with {participant})
        - adaptive_config_templates: Dict[str, str] (path templates with {participant} and {device})
        - parameter_field_in_template: str (field path with {NREMState})
        - nrem_state_on_device: str (path template with {participant} and {device})
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing the paths of the shipped files
    """
    shipped_files = {}
    participant = participant_info.select("RCS#").unique().item()

    # Process each side (Left/Right)
    for side, template_path in config["adaptive_config_templates"].items():

        # Skip if side not in parameters, e.g. unilateral participants
        if side not in parameters.keys():
            continue

        # Get device ID (participant + first letter of side)
        device = participant + side[0]
        
        # Load template
        template = _load_template(Path(template_path.format(
            participant=participant,
            device=device
        )))

        nrem_state_mapping = _load_template(Path(config["nrem_state_on_device"].format(
            participant=participant,
            device=device
        )))

        if "NREM" not in list(nrem_state_mapping.keys()):
            nrem_state_mapping = bidict.bidict(nrem_state_mapping).inverse
            assert "NREM" in list(nrem_state_mapping.keys()), "NREM state not found in nrem_state_mapping"
        
        # Create parameter mapping for this side
        parameter_mapping = {}
        field_path = config["parameter_field_in_template"].format(NREMState=nrem_state_mapping["NREM"]).replace(" ", "")
        parameter_mapping[side] = field_path
        
        # Update template with parameters
        updated_template = _map_parameters_to_template(
            template=template,
            parameters=parameters,
            parameter_mapping=parameter_mapping
        )
        
        # Write updated template to destination
        destination_path = Path(config["shipment_destination_path"].format(
            participant=participant
        )) / f"adaptive_config_{side[0]}.json"
        
        # Ensure destination directory exists
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write updated template
        with open(destination_path, 'w') as f:
            json.dump(updated_template, f, indent=4)
        
        shipped_files[side] = str(destination_path)
    
    return shipped_files


def update_current_target_amp_cache(parameters: Dict[str, Any], participant_info: pl.DataFrame, **config) -> Dict[str, Any]:
    """
    Update the current target amp cache on the device.
    
    Parameters
    ----------
    parameters : Dict[str, Any]
        Dictionary containing the parameters to ship
    config : Dict[str, Any]
        Configuration containing:
        - cache_path: str (path template with {participant})
        - cache_field: str (field in cache to update)
    
    Returns
    -------
    """
    participant = participant_info.select("RCS#").unique().item()
    
    # Format the cache path with the device ID
    cache_path = Path(config["cache_path"].format(participant=participant))
    
    # Ensure the directory exists
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if cache file exists, if so load it, otherwise create new
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
        except json.JSONDecodeError:
            # Handle case where file exists but is not valid JSON
            cache_data = {}
    else:
        cache_data = {}
    
    # Update cache with parameters
    cache_data.update(parameters)
    
    # Write updated cache to file
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=4)
    
    return {"cache_path": str(cache_path)}