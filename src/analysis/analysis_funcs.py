import polars as pl
from scipy import signal
import numpy as np
import polars.selectors as cs
from typing import List
import numpy.typing as npt
from scipy.signal import ShortTimeFFT


# ----- Quality Assurance Analysis Functions ------
def null_ratio(data: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """
    Calculate the percentage of null values for specified columns in a DataFrame.

    Args:
        data (pl.DataFrame): The input DataFrame to analyze.
        columns (list[str]): A list of column names to check for null values.

    Returns:
        pl.DataFrame: A DataFrame containing the percentage of null values for each specified column.
    """
    return data.select(
        (pl.col(*columns).null_count() / pl.col(*columns).len()).name.suffix(
            "_null_ratio"
        )
    )


def recording_time(data: pl.DataFrame, time_col: str = "localTime") -> pl.DataFrame:
    """
    Calculate the recording time for specified columns in a DataFrame.
    time_col is assumed to be a column of type pl.Datetime.
    """
    return data.select(
        (pl.col(time_col).max() - pl.col(time_col).min())
        .dt.total_minutes()
        .alias("total_recording_time_minutes"),
        pl.col(time_col).min().alias("Start_Time"),
        pl.col(time_col).max().alias("End_Time"),
    )


def get_session_numbers(
    data: pl.DataFrame, session_col: str = "SessionNumber", device_col: str = "Device"
) -> pl.DataFrame:
    """
    Get the unique session numbers from data.
    """
    return data.select(
        pl.col(session_col).unique().str.join(", ").alias("SessionNumber")
    )


def rcs_cDBS_qa_analysis(data: pl.DataFrame, kwargs: dict) -> pl.DataFrame:
    """
    Calculate the quality assurance analysis for specified columns in a DataFrame.
    """
    return pl.concat(
        [
            get_session_numbers(data),
            recording_time(data),
            null_ratio(data, ["^TD_key.*$"]),
        ],
        how="horizontal",
    )


def rcs_aDBS_qa_analysis(data: pl.DataFrame, kwargs: dict) -> pl.DataFrame:
    """
    Calculate the quality assurance analysis for specified columns in a DataFrame.
    """
    return pl.concat(
        [
            get_session_numbers(data),
            recording_time(data),
            time_in_each_stim_amplitude(data),
            null_ratio(data, ["^TD_key.*$"]),
        ],
        how="horizontal",
    )


# ----- Time Domain Analysis Functions ------
def time_in_each_stim_amplitude(
    data: pl.DataFrame, time_col: str = "localTime", stim_col: str = "stimAmplitude"
) -> pl.DataFrame:
    """
    Calculate the time in each stim amplitude for specified columns in a DataFrame.
    """
    return data.group_by(stim_col).agg(
        pl.col(time_col)
        .diff()
        .dt.total_seconds()
        .sum()
        .alias("time_in_each_stim_amplitude")
    )


# ----- Frequency Domain Analysis Functions ------
def get_psd_polars(
    df: pl.DataFrame,
    td_columns=[],
    freq_range=[0.5, 100],
    sampling_frequency=500,
    window_size=1024,
    noverlap=512,
    log=True,
    epoch=True,
    epoch_kwargs: dict = {},
):
    """
    Calculate the power spectral density (PSD) for each time domain column in the DataFrame.

    Args:
        df (pl.DataFrame): The DataFrame containing the (already epoched and cleaned)time domain data.
        td_columns (list): List of column names in the DataFrame that contain the time domain data.
        freq_range (list, optional): The frequency range for calculating the PSD. Defaults to [0.5, 100].
        sampling_frequency (int, optional): The sampling frequency of the time domain data. Defaults to 500.
        window_size (int, optional): The size of the window used for calculating the PSD. Defaults to 1024.
        noverlap (int, optional): The number of samples to overlap between windows. Defaults to 512.
        log (bool, optional): Whether to apply logarithm to the PSD values. Defaults to True.

    Returns:
        pl.DataFrame: The DataFrame with additional columns for the PSD values and frequency values.
    """

    if epoch:
        df = epoch_df_by_timesegment(df, **epoch_kwargs)

    # Assumes each time domain column is already epoched (e.g. time_domain_base.epoch_data was called)
    for col in td_columns:
        # Calculate PSD
        f, pxx = signal.welch(
            df.get_column(col).to_numpy(),
            fs=sampling_frequency,
            nperseg=window_size,
            noverlap=noverlap,
            axis=-1,
        )

        # Select PSD values within desired frequency range
        pxx = pxx[:, np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]]

        if log:
            pxx = np.log10(pxx)

        # Add PSD values to DataFrame
        df = df.with_columns(
            pl.Series(
                name=f"{col}_psd",
                values=pxx,
                dtype=pl.Array(inner=pl.Float32, width=pxx.shape[1]),
            )
        )

    # Add frequency values to DataFrame. Entire column will be an identical vector of frequency values.
    f_mat = np.tile(
        f[np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]], (df.height, 1)
    )
    df = df.with_columns(
        pl.Series(
            name=f"psd_freq",
            values=f_mat,
            dtype=pl.Array(inner=pl.Float32, width=f_mat.shape[1]),
        )
    )

    return df.select(pl.col("^.*psd.*$"))


def get_spectrograms_polars(
    df: pl.DataFrame, td_columns: list[str], **kwargs: dict
) -> pl.DataFrame:
    """
    Calculate the spectrogram for each time domain column in the DataFrame.
    """
    specs = {}
    specs["localTime"] = (
        df.select(pl.col("localTime").gather_every(kwargs.get("hop", 250)))
        .to_numpy()
        .flatten()
    )
    for col in td_columns:
        spectrogram = get_spectrograms(
            df.get_column(col).fill_null(0).to_numpy(), **kwargs
        ).T  # Transpose to match timestamps as rows
        if spectrogram.shape[0] != len(specs["localTime"]):
            spectrogram = spectrogram[
                : len(specs["localTime"]), :
            ]  # Trim to match lengths, sometimes spectrogram is slightly longer than localTime (by one hop)
        specs[f"{col}_spectrogram"] = spectrogram

    return pl.DataFrame(specs).with_columns(
        pl.col("localTime").dt.convert_time_zone("America/Los_Angeles")
    )  # Converting to numpy messes with time zone. Convert back to Los Angeles time.


def get_spectrograms(
    X: npt.NDArray,
    win_size=500,
    win_type="hamming",
    win_params={},
    hop=250,
    fs=500,
    scale_to="psd",
    axis=-1,
    log=True,
    freq_ranges=[[0.5, 100]],
):
    """
    Calculate the spectrogram of the input data.

    Args:
        X (np.ndarray): Input data for spectrogram calculation.
        win (int): Size of the window used for spectrogram calculation.
        hop (int): Number of samples to overlap between windows.
        fs (int): Sampling frequency of the data.
        scale_to (str): Scaling method for spectrogram calculation.
        axis (int): Axis along which the spectrogram is calculated.
        log (bool): Whether to apply logarithmic scaling to the spectrogram.
        freq_ranges (list of lists of floats): List of frequency ranges to keep in the spectrogram. Should be k x 2, where k is the number of frequency ranges to keep.

    Returns:
        np.ndarray: Input data, where specified axis is converted to spectrogram.
    """
    win = signal.get_window(win_type, win_size, **win_params)
    sft = ShortTimeFFT(win=win, hop=hop, fs=fs, scale_to=scale_to)
    Sxx = sft.spectrogram(X, axis=axis)
    Sxx = np.clip(Sxx, 1e-10, None)  # Clip to avoid log(0) errors

    if log:
        Sxx = np.log10(Sxx)

    if freq_ranges:
        freq_inds_to_keep = np.concatenate(
            [
                np.where((sft.f >= freq_range[0]) & (sft.f <= freq_range[1]))[0]
                for freq_range in freq_ranges
            ],
            axis=-1,
        )

        if axis == -1:
            Sxx = np.take(
                Sxx, freq_inds_to_keep, axis=-2
            )  # The second to last axis is the frequency axis
        else:
            Sxx = np.take(Sxx, freq_inds_to_keep, axis=axis)

    return Sxx


# ----- Time-Frequency Domain Analysis Functions ------


# ----- Signal Analysis Functions ------


# ----- Data Cleaning Functions ------
def epoch_df_by_timesegment(
    df: pl.DataFrame,
    interval: str = "1s",
    period: str = "2s",
    sample_rate: int = 500,
    align_with_PB_outputs: bool = False,
    td_columns: List[str] = ["TD_key0", "TD_key2", "TD_key3"],
    drop_nulls_in_td_columns_before_epoching: bool = True,
    sort_by_col="localTime",
    group_by_cols: List[str] = [],
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
    - td_columns (List[str]): A list of raw time domain columns to include in the resulting DataFrame. Default is ['TD_key0', 'TD_key2', 'TD_key3'].
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
