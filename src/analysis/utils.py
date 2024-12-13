import polars as pl
from typing import List


def epoch_df_by_timesegment(
    df: pl.DataFrame,
    interval: str = "1s",
    period: str = "2s",
    sample_rate: int = 500,
    align_with_PB_outputs: bool = False,
    td_columns: List[str] = ["TD_BG", "TD_key2", "TD_key3"],
    drop_nulls_in_td_columns_before_epoching: bool = True,
    sort_by_col="localTime",
    group_by_cols: List[str] = ["SessionIdentity"],
    scalar_cols: List[str] = [],
    vector_cols: List[str] = [],
) -> pl.DataFrame:
    """
    Epoch a DataFrame based on a time interval and period.

    Parameters:
    - df (polars.DataFrame): The DataFrame to be epoched.
    - interval (str): The time interval between the start of each time segment in seconds. Default is '1s'.
    - period (str): The length of each time segment in seconds. Default is '2s'.
    - sample_rate (int): The sampling rate of the data. Used to calculate the number of samples in each time segment. Default is 500.
    - align_with_PB_outputs (bool): If True, the time segments will be aligned with the Power Band outputs. Default is False.
    - td_columns (List[str]): A list of raw time domain columns to include in the resulting DataFrame. Default is ['TD_BG', 'TD_key2', 'TD_key3'].
    - drop_nulls_in_td_columns_before_epoching (bool): If True, rows where any of the specified time domain columns are null will be dropped before epoching. Default is True.
    - sort_by_cols (str): Column by which windowing is performed. Default is 'localTime'. Needs to be a datetime column.
    - group_by_cols (List[str]): A list of columns to group by. Default is ['SessionIdentity'].
    - scalar_cols (List[str]): A list of columns to include in the resulting DataFrame, where a single scalar value, the last value in the aggregation, is extracted after epoching. Default is [].
    - vector_cols (List[str]): A list of columns to include in the resulting DataFrame, where the aggregation creates a vector for the column values within each epoched window. Default is [].
    # TODO: Consider including kwarg that is a list of functions to apply to column subset, e.g. [pl.col(col).mean().alias(f'{col}_mean') for col in td_columns]

    Returns:
    - polars.DataFrame: A new DataFrame with the specified columns and epoched time segments.
    """

    # TODO: Consider 'streaming' option to save on RAM

    td_cols = cs.by_name(*td_columns)
    if drop_nulls_in_td_columns_before_epoching:
        df_filtered = df.filter(
            pl.all_horizontal(td_cols.is_not_null())
            & pl.all_horizontal(td_cols.is_not_nan())
        )
    else:
        df_filtered = df

    if align_with_PB_outputs:
        df_pb_count = (
            df_filtered.join(
                df_filtered.filter(pl.col("Power_Band8").is_not_null())
                .select("DerivedTime")
                .with_row_count(),
                on="DerivedTime",
                how="left",
            )
            .with_columns(pl.col("row_nr").fill_null(strategy="backward"))
            .rename({"row_nr": "PB_count"})
        )

        num_windows_in_each_period = int(period[:-1]) // int(interval[:-1])
        df_pb_count = df_pb_count.with_columns(
            [
                pl.when((pl.col("PB_count") % num_windows_in_each_period) != i)
                .then(pl.lit(None))
                .otherwise(pl.col("PB_count"))
                .fill_null(strategy="backward")
                .alias(f"PB_count_{i}")
                for i in range(num_windows_in_each_period)
            ]
        )

        # NOTE: Windows are likely not in chronological order
        df_epoched = (
            pl.concat(
                [
                    df_pb_count.group_by(group_by_cols + [f"PB_count_{i}"])
                    .agg(
                        [pl.col(td_col) for td_col in td_columns]
                        + [pl.col(td_columns[0]).count().alias("TD_count")]
                        + [pl.col(col) for col in vector_cols]
                        + [pl.col(col).drop_nulls().first() for col in scalar_cols]
                    )
                    .rename({f"PB_count_{i}": "PB_ind"})
                    for i in range(num_windows_in_each_period)
                ],
                how="vertical",
            )
            .select(pl.all().shrink_dtype())
            .rechunk()
        )

    else:
        epoch_length = int(period[:-1]) * sample_rate
        df_epoched = (
            df_filtered.sort(sort_by_col)
            .group_by_dynamic(
                sort_by_col, every=interval, period=period, by=group_by_cols
            )
            .agg(
                [pl.col(td_col) for td_col in td_columns]
                + [
                    pl.col(td_col).count().name.suffix("_TD_count")
                    for td_col in td_columns
                ]
                + [pl.col(col).name.suffix("_vec") for col in vector_cols]
                + [pl.col(col).drop_nulls().last() for col in scalar_cols]
            )
            .select(pl.all().shrink_dtype())
        )

        df_epoched = (
            df_epoched.with_columns(
                [
                    pl.col(td_col)
                    .list.eval(pl.element().is_not_null())
                    .list.all()
                    .name.suffix("_contains_no_null")
                    for td_col in td_columns
                ]
                # Remove rows where the TD data is null, or where the TD data is not the correct length
            )
            .filter(
                # Remove rows where the TD data is null or not the correct length
                (pl.all_horizontal(pl.col("^.*_TD_count$") == epoch_length))
                & (pl.all_horizontal("^.*_contains_no_null$"))
            )
            .with_columns(
                [
                    pl.col(col).cast(pl.Array(width=epoch_length, inner=pl.Float64))
                    for col in td_columns
                ]
            )
            .select(pl.all().exclude("^.*TD_count$"))
            .select(pl.all().exclude("^.*_contains_no_null$"))
        )

    if df_epoched.height == 0:
        raise ValueError(
            "Epoched DataFrame is empty. Check that the specified columns are present in the DataFrame, and that the specified interval, period, and sample rate are valid."
        )

    return df_epoched
