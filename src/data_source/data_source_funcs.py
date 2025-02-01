import polars as pl
import os


def get_rcs_data(session: pl.DataFrame, file_name: str) -> pl.DataFrame:
    """Get data from the RCS data source."""
    return pl.read_parquet(os.path.join(session["session_report_path"], file_name))
