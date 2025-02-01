import polars as pl


def check_data_completeness(data: pl.DataFrame, min_completeness: float = 0.5) -> bool:
    """
    Check if data has sufficient non-null values.
    Returns True if check fails (insufficient data), False if passes.
    """
    completeness = 1 - (data.null_count() / len(data))
    return completeness < min_completeness
