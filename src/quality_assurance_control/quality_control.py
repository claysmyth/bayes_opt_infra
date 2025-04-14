import polars as pl
from typing import Dict


def check_data_completeness(data: pl.DataFrame, min_completeness: float = 0.5) -> bool:
    """
    Check if data has sufficient non-null values.
    Returns True if check fails (insufficient data), False if passes.
    """
    completeness = 1 - (data.null_count() / len(data))
    return completeness < min_completeness


def check_data_length(
    data: pl.DataFrame, sample_rate: int, min_duration_hours: int = 4
) -> bool:
    """
    Check if data has sufficient length.
    Returns True if check fails (insufficient data), False if passes.
    """
    data_length = len(data)
    min_length = sample_rate * 60 * 60 * min_duration_hours
    return data_length < min_length


def check_bilateral_data_completeness(
    bilateral_data: Dict[str, pl.DataFrame],
    qc_column: str = 'Power_Band5',
    min_duration_hours: float = 2.0,
    sample_rate: float = 1/15
) -> bool:
    """
    Check if bilateral RCS data has sufficient non-null values and duration.
    Returns True if check fails (insufficient data), False if passes.
    
    Parameters
    ----------
    bilateral_data : Dict[str, pl.DataFrame]
        Dictionary containing 'Left' and 'Right' DataFrames from get_bilateral_rcs_data
    qc_column : str
        Column name to check for non-null values
    min_duration_hours : float, optional
        Minimum required duration in hours, by default 2.0
    sample_rate : float, optional
        Sample rate in Hz, by default 1/15 (Once per 15 seconds). This default is because we update delta power into the Linear Discriminant every 15 seconds.
        
    Returns
    -------
    bool
        True if check fails (insufficient data), False if passes
    """
    # Calculate minimum required samples
    min_samples = sample_rate * 60 * 60 * min_duration_hours
    
    # Check each hemisphere
    for side, data in bilateral_data.items():
        # Filter non-null rows for the specified column
        non_null_data = data.filter(pl.col(qc_column).is_not_null())
        
        # Check data length based on non-null values
        if len(non_null_data) < min_samples:
            print(f"{side} hemisphere has insufficient non-null {qc_column} values ({len(non_null_data)} samples) for minimum duration ({min_samples} samples required)")
            return True
    
    return False
