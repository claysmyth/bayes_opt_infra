import polars as pl
from typing import Dict
import json
import warnings


def check_rcs_data_completeness(data: pl.DataFrame, _: pl.DataFrame, **config: dict) -> bool:
    # TODO: Check both left and right... sum should be >= 2 hours
    """
    Check if data has sufficient non-null values.
    Returns True if check fails (insufficient data), False if passes.
    
    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing device data
    _ : pl.DataFrame
        DataFrame containing session information
    config : dict
        Configuration dictionary containing:
        - min_completeness: float (hours) # Data sum across both hemispheres need to be >= min_completeness to be considered "good enough".
            If only one hemisphere, require half the completeness.
        - sample_rate: float (Hz) # Default to once per 15 seconds
        
        Deprecated. If we want to filter on target amplitude, we need the following:
        - nrem_state_base_path: str (path template with {participant} and {device} placeholders)
        - current_adaptive_config_path: str (path template with {participant} and {device} placeholders)
    """
    SECONDS_PER_HOUR = 60 * 60
    # Get participant and device info from data
    data_rec_lengths = {}
    for i, (side, data) in enumerate(data.items()):
        # device = participant + side[0]
        # device = data.select("device").item()
        
        # # Load NREM state mapping
        # nrem_state_path = config["nrem_state_base_path"].format(
        #     participant=participant,
        #     device=device
        # )
        # try:
        #     with open(nrem_state_path, 'r') as f:
        #         nrem_state_mapping = json.load(f)
        # except Exception as e:
        #     warnings.warn(f"Failed to load NREM state mapping from {nrem_state_path}: {str(e)}")
        #     return True
        
        # # Load current adaptive config
        # current_params_path = config["current_target_amps_path"].format(
        #     participant=participant,
        #     device=device
        # )
        # try:
        #     with open(current_params_path, 'r') as f:
        #         current_params = json.load(f)
        # except Exception as e:
        #     warnings.warn(f"Failed to load current params from {current_params_path}: {str(e)}")
        #     return True

        side_length = data.filter(
            pl.col("Adaptive_CurrentAdaptiveState").is_not_null()
        ).height

        sample_rate = config.get("sample_rate", 1)
        length_in_hours = side_length / sample_rate / SECONDS_PER_HOUR
        data_rec_lengths[side] = length_in_hours

        
    # Check data length
    # Determine minimum completeness requirement based on number of hemispheres
    min_completeness = config.get("min_completeness", 3)
    if len(data_rec_lengths) == 1:
        # If only one hemisphere, require half the completeness
        min_completeness = min_completeness / 2
    
    if sum(data_rec_lengths.values()) < min_completeness:
        warnings.warn(f"Data length ({sum(data_rec_lengths.values())} hours) is less than required minimum ({min_completeness} hours)")
        return True
    
    return False


# def check_data_length(data: pl.DataFrame, config: dict) -> bool:
#     """
#     Check if data has sufficient length.
#     Returns True if check fails (insufficient data), False if passes.
    
#     Parameters
#     ----------
#     data : pl.DataFrame
#         DataFrame containing device data
#     config : dict
#         Configuration dictionary containing:
#         - sample_rate: int (Hz)
#         - min_duration_hours: int
#     """
#     sample_rate = config.get("sample_rate", 1/15)  # Default to once per 15 seconds
#     min_duration_hours = config.get("min_duration_hours", 4)
    
#     data_length = len(data)
#     min_length = sample_rate * 60 * 60 * min_duration_hours
#     return data_length < min_length


# def check_bilateral_data_completeness(data: pl.DataFrame, config: dict) -> bool:
#     """
#     Check if bilateral RCS data has sufficient non-null values and duration.
#     Returns True if check fails (insufficient data), False if passes.
    
#     Parameters
#     ----------
#     data : pl.DataFrame
#         DataFrame containing device data
#     config : dict
#         Configuration dictionary containing:
#         - qc_column: str (column name to check for non-null values)
#         - min_duration_hours: float
#         - sample_rate: float
#     """
#     # Get configuration parameters with defaults
#     qc_column = config.get("qc_column", "Power_Band5")
#     min_duration_hours = config.get("min_duration_hours", 2.0)
#     sample_rate = config.get("sample_rate", 1/15)  # Default to once per 15 seconds
    
#     # Calculate minimum required samples
#     min_samples = sample_rate * 60 * 60 * min_duration_hours
    
#     # Check each hemisphere
#     for side, data in data.group_by("device").agg(pl.all()):
#         # Filter non-null rows for the specified column
#         non_null_data = data.filter(pl.col(qc_column).is_not_null())
        
#         # Check data length based on non-null values
#         if len(non_null_data) < min_samples:
#             warnings.warn(f"{side} hemisphere has insufficient non-null {qc_column} values ({len(non_null_data)} samples) for minimum duration ({min_samples} samples required)")
#             return True
    
#     return False
