import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from datetime import timedelta
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import polars.selectors as cs
import json
import bidict
from pathlib import Path
import copy
def extrapolate_column(
    df: pl.DataFrame, 
    column_name: str = 'Power_Band5',
    extrapolation_duration_hours: float = 2.0,
    num_seconds_for_power_averaging: int = 15
) -> pl.DataFrame:
    """
    Extrapolate values in a specified column using LinearRegression and extend the dataframe.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing the column to extrapolate and 'localTime' column
    column_name : str, optional
        Name of the column to extrapolate, by default 'Power_Band5'
    extrapolation_duration_hours : float, optional
        Duration in hours to extrapolate, by default 2.0
    num_seconds_for_power_averaging : int, optional
        Number of seconds between data points, by default 15
        
    Returns
    -------
    pl.DataFrame
        Extended dataframe with extrapolated values
    """
    # Calculate time since start in seconds
    first_timestamp = df['localTime'].min()
    df = df.with_columns(
        (pl.col('localTime') - first_timestamp).dt.total_seconds().alias('Time_Since_Start_Seconds')
    )
    
    # Prepare data for regression
    X = df['Time_Since_Start_Seconds'].to_numpy().reshape(-1, 1)
    y = df[column_name].to_numpy()
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate time points for extrapolation
    current_max_time = df['Time_Since_Start_Seconds'].max()
    extrapolation_seconds = extrapolation_duration_hours * 3600
    new_times = np.arange(
        current_max_time + num_seconds_for_power_averaging,
        current_max_time + extrapolation_seconds + num_seconds_for_power_averaging,
        num_seconds_for_power_averaging
    )
    
    # Generate predictions
    new_predictions = model.predict(new_times.reshape(-1, 1))
    
    # Create new timestamps
    new_timestamps = pl.Series(
        [first_timestamp + timedelta(seconds=float(t)) for t in new_times],
        dtype=pl.Datetime(time_unit='us', time_zone='America/Los_Angeles')
    )
    
    # Create extrapolated dataframe
    extrapolated_df = pl.DataFrame({
        'localTime': new_timestamps,
        column_name: new_predictions,
        'Time_Since_Start_Seconds': new_times,
        'is_extrapolated': [True] * len(new_times)
    }).with_columns(pl.col(column_name).clip(0, None))
    
    # Add extrapolation flag to original data
    df = df.with_columns(
        pl.lit(False).alias('is_extrapolated')
    )
    
    # Combine original and extrapolated data
    result_df = pl.concat([
        df,
        extrapolated_df,
    ], how='diagonal_relaxed').sort('localTime')
    
    return result_df

def average_over_time_segments(df: pl.DataFrame, time_column: str, columns_to_average: List[str], interval: str, period: str, group_by: List[str] = []) -> pl.DataFrame:
    """
    Average over time segments.
    """
    cols = cs.by_name(*columns_to_average)
    df_averaged = (
        df.sort(time_column)
        .group_by_dynamic(
            time_column, every=interval, period=period, group_by=group_by
        )
        .agg(
            cols.mean()
        )
    )
    return df_averaged


def add_running_average(df: pl.DataFrame, column_name: str) -> pl.DataFrame:
    """
    Add a running average column to the dataframe.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe
    column_name : str
        Name of the column to calculate running average for
        
    Returns
    -------
    pl.DataFrame
        Dataframe with added running average column
    """
    return df.with_row_index(offset=1).with_columns(
        (pl.col(column_name).cum_sum() / pl.col('index')).alias(f'{column_name}_running_average')
    ).drop('index')


def plot_single_night_power_band(
    df: pl.DataFrame,
    power_column: str = 'Power_Band5',
    title: str = None,
    max_x_axis: int = 40_000,
    extrapolation_duration_hours: float = 2.0,
    regression_line: bool = False,
    **kwargs
) -> plt.Figure:
    """
    Plot power band data for a single night with running average and regression line using matplotlib.
    
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing power band and localTime columns
    power_column : str, optional
        Name of the power band column to plot, by default 'Power_Band5'
    title : str, optional
        Title for the plot, by default None
    max_x_axis : int, optional
        Maximum x-axis value in seconds, by default 40_000
    extrapolation_duration_hours : float, optional
        Duration in hours to extrapolate, by default 2.0
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Add running average
    df = average_over_time_segments(
        df=df,
        time_column='localTime',
        columns_to_average=[power_column],
        interval='15s',
        period='15s',
        group_by=['SessionNumber']
    )
    
    df = add_running_average(df, power_column)
    
    # Calculate time since start
    first_timestamp = df['localTime'].min()
    df = df.with_columns(
        (pl.col('localTime') - first_timestamp).dt.total_seconds().alias('Time_Since_Start_Seconds')
    )
    
    # Convert to pandas for matplotlib
    df_pd = df.to_pandas()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')
    
    # Plot power band
    ax.plot(
        df_pd['Time_Since_Start_Seconds'],
        df_pd[power_column],
        label=power_column,
        color='blue',
        linewidth=1
    )
    
    # Plot running average
    ax.plot(
        df_pd['Time_Since_Start_Seconds'],
        df_pd[f'{power_column}_running_average'],
        label=f'{power_column} Running Average',
        color='purple',
        linewidth=2
    )
    
    # Add average line
    avg_power = df_pd[power_column].mean()
    ax.axhline(
        y=avg_power,
        color='green',
        linestyle='-',
        label=f'Avg: {avg_power:.2f}'
    )
    
    # Add regression line if enough points
    if regression_line and len(df_pd) > 1:
        x = df_pd['Time_Since_Start_Seconds'].values.reshape(-1, 1)
        y = df_pd[power_column].values
        model = LinearRegression()
        model.fit(x, y)
        x_line = np.array([0, max_x_axis])
        y_line = model.predict(x_line.reshape(-1, 1))
        ax.plot(
            x_line,
            y_line,
            color='red',
            linestyle='--',
            label=f'Slope: {model.coef_[0]:.2f}'
        )
    
    # Set labels and title
    ax.set_xlabel('Time Since Recording Start (seconds)')
    ax.set_ylabel(power_column)
    ax.set_xlim(0, max_x_axis)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{power_column} Over Time')
    
    # Show legend
    ax.legend(frameon=False)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_session_against_baseline_aggregate(
    df: pl.DataFrame,
    start_time_seconds: int = 3600,
    max_x_axis: int = 40_000,
    nrem_state_base_path: str = None,
    current_target_amps_path: str = None,
    baseline_running_stats_path: str = None,
    session_info: pl.DataFrame = None,
    title: str = None,
) -> plt.Figure:
    """
    Create an aggregate plot showing baseline sessions, mean, std, and current session NREM data using matplotlib.
    """
    # Extract session number and filter session_info
    session_number = df.get_column("SessionNumber").unique().first()
    session_info = session_info.filter(pl.col("Session#").str.contains(session_number))
    participant = session_info.get_column("RCS#").unique().item()
    device = session_info.get_column("Device").unique().item()

    # Load the NREM state mapping from the provided path
    with open(Path(nrem_state_base_path.format(
        participant=participant,
        device=device
    )), 'r') as f:
        nrem_state_mapping = json.load(f)
    
    if "NREM" not in list(nrem_state_mapping.keys()):
        nrem_state_mapping = bidict.bidict(nrem_state_mapping).inv
        assert "NREM" in list(nrem_state_mapping.keys()), "NREM state not found in nrem_state_mapping"

    # Load the current target amps from the provided path
    with open(Path(current_target_amps_path.format(participant=participant)), 'r') as f:
        current_target_amps = json.load(f)

    # Load the baseline running stats from the provided path
    baseline_running_stats = pl.read_parquet(baseline_running_stats_path.format(participant=participant, device=device))

    # Filter the data df for NREM state and target amp
    nrem_delta_power = df.filter(
        # Filter for NREM state
        (pl.col("Adaptive_CurrentAdaptiveState") == nrem_state_mapping["NREM"])
        # Select the columns of interest
        ).select(
            pl.col('localTime'),
            pl.col("Adaptive_Ld0_featureInputs").list.get(0)
        # Filter out duplicate values of first Ld0. Each FFT cycle prints out identical values until a new updateRate cycle is reached.
        ).filter(
            pl.col("Adaptive_Ld0_featureInputs").diff().abs() != 0
        # Add a column for time since start
        ).with_columns(
            (pl.col("localTime") - pl.col("localTime").min()).dt.total_seconds().cast(pl.Float64).alias("Time_Since_Start_Seconds")
    )

    nrem_delta_power = add_running_average(nrem_delta_power, "Adaptive_Ld0_featureInputs")

    # Create line plots of Time_Since_Start_Seconds vs Mean and individual sessions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Get all session columns
    session_columns = [col for col in baseline_running_stats.columns if "Session" in col]

    # Create line plots of Time_Since_Start_Seconds vs Mean and individual sessions

    # Sort by time to ensure proper line plotting (using Polars)
    plot_data = baseline_running_stats.sort("Time_Since_Start_Seconds")


    # Create the first plot (full range)
    # Plot mean with thicker line
    time_data = plot_data.get_column("Time_Since_Start_Seconds").to_numpy()
    mean_data = plot_data.get_column("Mean").to_numpy()
    std_data = plot_data.get_column("Std").to_numpy()

    # Plot mean with standard deviation as shaded region
    ax1.plot(time_data, mean_data, linewidth=3, color='black', label='Mean')
    ax1.fill_between(time_data, mean_data - std_data, mean_data + std_data, 
                    color='gray', alpha=0.15, label='±1 Std Dev')


    # Plot Power_Band5_running_average as bright blue line
    ax1.plot(nrem_delta_power.get_column("Time_Since_Start_Seconds").to_numpy(),
            nrem_delta_power.get_column("Adaptive_Ld0_featureInputs_running_average").to_numpy(),
            linewidth=2, color='blue', label='Power Band 5 Running Average')

    # Plot each session with semi-transparent lines for comparison
    for session in session_columns:
        ax1.plot(plot_data.get_column("Time_Since_Start_Seconds"), 
                plot_data.get_column(session), 
                linewidth=1, alpha=0.5, label=session)
    ax1.set_title("Average Power Band 5 Over Time (Full Range)")
    ax1.set_xlabel("Time Since Start (Seconds)")
    ax1.set_ylabel("Power Band 5")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(right=40_000)  # Set maximum x-axis limit to 40,000

    # Create the second plot (starting at x=3600)
    filtered_data = plot_data.filter(pl.col("Time_Since_Start_Seconds") >= 3600)
    filtered_time = filtered_data.get_column("Time_Since_Start_Seconds").to_numpy()
    filtered_mean = filtered_data.get_column("Mean").to_numpy()
    filtered_std = filtered_data.get_column("Std").to_numpy()

    # Plot mean with standard deviation as shaded region
    ax2.plot(filtered_time, filtered_mean, linewidth=3, color='black', label='Mean')
    ax2.fill_between(filtered_time, filtered_mean - filtered_std, filtered_mean + filtered_std, 
                    color='gray', alpha=0.15, label='±1 Std Dev')


    # Plot Power_Band5_running_average as bright blue line (filtered to match the time range)
    filtered_nrem_delta = nrem_delta_power.filter(pl.col("Time_Since_Start_Seconds") >= 3600)
    ax2.plot(filtered_nrem_delta.get_column("Time_Since_Start_Seconds").to_numpy(),
            filtered_nrem_delta.get_column("Adaptive_Ld0_featureInputs_running_average").to_numpy(),
            linewidth=2, color='blue', label='Power Band 5 Running Average')


    # Plot randomly selected sessions with thinner, semi-transparent lines
    for session in session_columns:
        ax2.plot(filtered_data.get_column("Time_Since_Start_Seconds"), 
                filtered_data.get_column(session), 
                linewidth=1, alpha=0.5, label=session)
    ax2.set_title("Average Cortical Delta Power Over Time (After 1 Hour)")
    ax2.set_xlabel("Time Since Start (Seconds)")
    ax2.set_ylabel("Cortical Delta Power")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(right=40_000)  # Set maximum x-axis limit to 40,000

    # Add legend to a separate figure area to avoid cluttering
    plt.figlegend(loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4, fontsize='small')

    # Add some styling
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Make room for the legend at the bottom

    return fig


def plot_adaptive_state_and_amplitude(
    df: pl.DataFrame,
    nrem_state_base_path: str = None,
    title: str = None,
    figsize: tuple = (12, 8),
    session_info: pl.DataFrame = None,
    **kwargs
) -> plt.Figure:
    """
    Plot adaptive state and amplitude data over time using matplotlib.
    """
    # Extract session number and filter session_info
    session_number = df.get_column("SessionNumber").unique().first()
    session_info = session_info.filter(pl.col("Session#").str.contains(session_number))
    participant = session_info.get_column("RCS#").unique().item()
    device = session_info.get_column("Device").unique().item()

    # Load the NREM state mapping from the provided path
    with open(Path(nrem_state_base_path.format(
        participant=participant,
        device=device
    )), 'r') as f:
        nrem_state_mapping = json.load(f)

    # Prepare data
    filtered_df = df.filter(pl.col("Adaptive_CurrentAdaptiveState").is_not_null()).with_columns(
        pl.col("Adaptive_CurrentProgramAmplitudesInMilliamps").list.get(0).alias("Amplitude"),
        pl.when(pl.col("Adaptive_CurrentAdaptiveState") == "State 0").then(0).otherwise(1).alias("Adaptive_State")
    )
    min_time = filtered_df["localTime"].min()
    filtered_df = filtered_df.with_columns(
        (pl.col("localTime") - min_time).dt.total_seconds().alias("time_since_start")
    )
    df_pd = filtered_df.to_pandas()

    nrem_state_mapping_int = {0: "State 0", 1: "State 1"}
    nrem_state_mapping_int = {k: nrem_state_mapping[v] for k, v in nrem_state_mapping_int.items()}

    # Create matplotlib subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax_state, ax_amp = axes

    # Plot adaptive state
    ax_state.plot(
        df_pd["time_since_start"],
        df_pd["Adaptive_State"],
        color='orange',
        linewidth=2,
        label="Adaptive State"
    )
    ax_state.set_ylabel("State")
    ax_state.set_yticks(list(nrem_state_mapping_int.keys()))
    ax_state.set_yticklabels(list(nrem_state_mapping_int.values()))
    ax_state.set_title("Adaptive State Over Time")
    ax_state.legend(frameon=False)
    ax_state.spines['top'].set_visible(False)
    ax_state.spines['right'].set_visible(False)

    # Plot amplitude
    ax_amp.plot(
        df_pd["time_since_start"],
        df_pd["Amplitude"],
        color='blue',
        linewidth=2,
        label="Amplitude (mA)"
    )
    ax_amp.set_xlabel("Time Since Start (seconds)")
    ax_amp.set_ylabel("Amplitude (mA)")
    ax_amp.set_title("Amplitude Over Time")
    ax_amp.legend(frameon=False)
    ax_amp.spines['top'].set_visible(False)
    ax_amp.spines['right'].set_visible(False)

    # Set overall title
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle("Adaptive State and Amplitude Over Time")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def experiment_contour_plot(experiment_tracker: 'ExperimentTracker', **kwargs) -> go.Figure:
    """
    Get the contour plot of the experiment.
    
    Parameters
    ----------
    experiment_tracker : ExperimentTracker
        Experiment tracker object
        
    Returns
    -------
    go.Figure: Plotly figure object with contour plot
    """

    return experiment_tracker.get_contour_plot_safe()


def trial_results(result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Save the results to a file.
    """
    return result


def sessions_info(result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Save the sessions info to a file.
    """
    return result


def experiment_snapshot(result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Save the experiment snapshot to a file.
    """
    return result