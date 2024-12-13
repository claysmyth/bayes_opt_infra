import polars as pl


def time_delta_from_first_session(session_df, data=None, max_time_delta=8):
    """
    If the start time of subsequent sessions is within the time delta of the first session, then those sessions are aggregated.
    session_df: polars DataFrame containing session information, formatted from 'Project' directory csv.
    data: list of polars DataFrames containing the raw data for each session. None here, because not used. To keep consistent with other aggregation criteria.
    max_time_delta: maximum time delta between first and subsequent sessions to be aggregated.
    """

    session_df = session_df.with_columns(
        (
            pl.col("TimeStarted").diff().dt.total_hours().cum_sum().cast(pl.Int32)
            // max_time_delta
        ).alias("Group")
    ).with_columns(pl.col("Group").fill_null(0))

    return session_df
