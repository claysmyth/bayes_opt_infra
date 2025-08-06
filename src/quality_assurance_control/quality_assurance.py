import polars as pl
import json
import re
from pathlib import Path


def check_stim_settings(data: pl.DataFrame, expected_amplitude: float = 1.0) -> bool:
    """
    Check if stimulation settings match expected values.
    Returns True if check fails (bad data), False if passes.
    """
    actual_amplitude = data["stim_amplitude"].mean()
    return abs(actual_amplitude - expected_amplitude) > 0.1


def check_rcs_sense_settings(_: pl.DataFrame, sessions_df, **config) -> bool:
    """
    Check if RCS device settings match expected values.
    Returns True if check fails (bad data), False if passes.
    
    Parameters
    ----------
    _ : pl.DataFrame
        DataFrame containing device data
    sessions_df : pl.DataFrame
        DataFrame containing session information
    config : dict
        Configuration dictionary containing:
        - settings_qa_dict: dict of expected settings
        - session_settings_csv_template_path: str (path template with {key} and {session})
        
    Returns
    -------
    bool
        True if check fails (bad data), False if passes
    """
    # Get device_id and session from data

    for session in sessions_df.iter_rows(named=True):
        device_id = session["RCS#"] + session["Side"][0]
        session = session["Session#"]

        print(f"Checking RCS sense settings for {device_id} and session {session}")

        # Get session settings
        session_settings_csv_path = config["session_settings_csv_template_path"].format(
            device=device_id, 
            session=session
        )
        session_settings_df = pl.read_csv(session_settings_csv_path)
        
        # Convert to dictionary for comparison
        session_settings = session_settings_df.to_dict(as_series=False)
        
        # Verify settings
        for key, value in config["settings_qa_dict"].items():
            if key not in session_settings.keys():
                print(f"{key} not found in session_settings_df")
                return True  # Check fails
            if session_settings[key] != value:
                print(f"{key} does not match expected value")
                print(f"Expected: {value}, Actual: {session_settings[key]}")
                return True  # Check fails
    
    return False  # All checks pass


def check_rcs_adaptive_settings(_: pl.DataFrame, sessions_df: pl.DataFrame, **config: dict) -> bool:
    """
    Check if RCS device adaptive settings match expected values from template.
    Returns True if check fails (bad data), False if passes.
    
    Parameters
    ----------
    _ : pl.DataFrame
        DataFrame containing device data
    sessions_df : pl.DataFrame
        DataFrame containing session information
    config : dict
        Configuration dictionary containing settings_qa_dict_filepath, 
        session_detector_settings_csv_template_path, and csv_to_template_field_mapping
        
    Returns
    -------
    bool
        True if check fails (bad data), False if passes
    """
    # Get device_id and session from data
    for session in sessions_df.iter_rows(named=True):
        participant = session["RCS#"]
        side = session["Side"][0]
        device_id = participant + side
        session = session["Session#"]

        print(f"Checking RCS adaptive settings for {device_id} and session {session}")
    
        # Load template settings
        template_path = config["settings_qa_dict_filepath"].format(
            participant=participant,
            device=device_id,
            side=side
        )
        with open(template_path, 'r') as f:
            template_settings = json.load(f)
        
        # Get session settings
        session_settings_csv_path = config["session_detector_settings_csv_template_path"].format(
            device=device_id,
            session=session
        )
        session_settings_df = pl.read_csv(session_settings_csv_path)
        
        # Get fractional fixed point value for scaling
        ffp_value = float(get_nested_value(template_settings, "Detection.LD0.FractionalFixedPointValue"))
        
        # Convert to dictionary for comparison
        session_settings = session_settings_df.to_dict(as_series=False)
        
        # Check each mapping
        for csv_key, template_path in config["csv_to_template_field_mapping"]["detector"].items():
            # Handle vector fields (those with *)
            if "*" in csv_key:
                base_key = csv_key.replace("_*", "")
                vector_template_value = get_nested_value(template_settings, template_path)
                
                # Check each vector element
                for i, template_val in enumerate(vector_template_value, start=1):
                    csv_key_with_index = f"{base_key}_{i}"
                    if csv_key_with_index not in session_settings:
                        print(f"{csv_key_with_index} not found in session settings")
                        return True
                    
                    csv_val = session_settings[csv_key_with_index][0]  # Get first element if list
                    
                    # Apply scaling for bias and vector values
                    if "bias" in csv_key.lower() or "vector" in csv_key.lower():
                        # csv_val = int(float(csv_val) * (2 ** ffp_value))
                        # Correct template value for scaling
                        template_val = int(template_val * (2 ** ffp_value)) & 0xFFFFFFFF
                    
                    if csv_val != template_val:
                        print(f"{csv_key_with_index} does not match expected value")
                        print(f"Expected: {template_val}, Actual: {csv_val}")
                        return True
            else:
                # Handle non-vector fields
                template_val = get_nested_value(template_settings, template_path)
                if csv_key not in list(session_settings.keys()):
                    print(f"{csv_key} not found in session settings")
                    return True
                
                csv_val = session_settings[csv_key][0]  # Get first element if list
                
                # Apply scaling for bias and vector values
                if "bias" in csv_key.lower() or "vector" in csv_key.lower():
                    # csv_val = int(float(csv_val) * (2 ** ffp_value))
                    # Correct template value for scaling
                    template_val = int(template_val * (2 ** ffp_value)) & 0xFFFFFFFF
                
                if csv_val != template_val:
                    print(f"{csv_key} does not match expected value")
                    print(f"Expected: {template_val}, Actual: {csv_val}")
                    return True
        
    return False  # All checks pass


def check_if_both_sides_are_present(_: pl.DataFrame, sessions_df: pl.DataFrame, **config: dict) -> bool:
    """
    Check if all sides in the current_target_amps_path are present in the sessions_df.
    Returns True if any side is missing, False if all are present.
    """

    participant = sessions_df["RCS#"].unique()[0]
    # Read the current_target_amps_path file
    with open(Path(config['current_target_amps_path'].format(participant=participant)), 'r') as f:
        target_amps = json.load(f)
    
    # Get the set of sides present in sessions_df
    sides_present = set(sessions_df.get_column('Side').to_list())
    
    # Check for each key in target_amps
    for side in target_amps.keys():
        if side not in sides_present:
            return True  # Missing side found
    
    return False  # All sides are present


def get_nested_value(d: dict, path: str) -> any:
    """Helper function to get nested dictionary values using dot notation."""
    keys = path.split('.')
    value = d
    for key in keys:
        value = value[key]
    return value