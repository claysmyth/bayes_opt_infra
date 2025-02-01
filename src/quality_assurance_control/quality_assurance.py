import polars as pl

def check_stim_settings(data: pl.DataFrame, expected_amplitude: float = 1.0) -> bool:
    """
    Check if stimulation settings match expected values.
    Returns True if check fails (bad data), False if passes.
    """
    actual_amplitude = data["stim_amplitude"].mean()
    return abs(actual_amplitude - expected_amplitude) > 0.1
