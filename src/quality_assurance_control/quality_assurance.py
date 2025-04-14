import polars as pl


def check_stim_settings(data: pl.DataFrame, expected_amplitude: float = 1.0) -> bool:
    """
    Check if stimulation settings match expected values.
    Returns True if check fails (bad data), False if passes.
    """
    actual_amplitude = data["stim_amplitude"].mean()
    return abs(actual_amplitude - expected_amplitude) > 0.1


def check_rcs_settings(data: pl.DataFrame, config: dict) -> bool:
    """
    Check if RCS device settings match expected values.
    Returns True if check fails (bad data), False if passes.
    
    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing device data
    config : dict
        Configuration dictionary containing settings_qa_dict and session_settings_csv_template_path
        
    Returns
    -------
    bool
        True if check fails (bad data), False if passes
    """
    # Get device_id and session from data
    device_id = data.select("device_id").item()
    session = data.select("session").item()
    
    # Get session settings
    session_settings_csv_path = config["session_settings_csv_template_path"].format(
        key=device_id, 
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