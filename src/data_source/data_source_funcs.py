import polars as pl
import os


def get_bilateral_rcs_data(sessions: pl.DataFrame, file_name: str = 'raw_data.parquet') -> dict:
    """
    Get data from the Bilateral ARCS data source.
    
    Returns a dictionary with 'Left' and 'Right' keys, each containing
    the corresponding side's data from the parquet file.
    """
    # Get unique session report paths
    unique_sessions = sessions.select('session_report_path', 'Side').unique()
    
    # Check that we have exactly two sides (Left and Right)
    if unique_sessions.height > 2:
        raise ValueError(f"Expected at most 2 unique sessions, but found {unique_sessions.height}")
    
    # Verify we have both Left and Right sides
    sides = unique_sessions['Side'].to_list()
    if 'Left' not in sides and 'Right' not in sides:
        raise ValueError(f"Missing required sides. Found: {sides}, need at least one 'Left' or one 'Right'")
    
    # Create dictionary with data for each side
    result = {}
    for row in unique_sessions.iter_rows(named=True):
        side = row['Side']
        path = row['session_report_path']
        result[side] = pl.read_parquet(os.path.join(path, file_name))
    
    return result
