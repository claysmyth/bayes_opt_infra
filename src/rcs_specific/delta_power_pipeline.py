"""
Pipeline to process and analyze delta power data from RCS devices.
This script processes RCS data to calculate and visualize delta power trends,
including extrapolation and cumulative averages, from the 'baseline' dataset for the sleep aDBS project.
The processed data is used for z-scoring new overnight recording session's average delta power. 

The script creates individual session plots and an overlay plot showing:
- Raw delta power values
- Cumulative averages
- Linear trends
- Time delta distributions

Example usage:
    # Basic usage with default config:
    python src/rcs_specific/delta_power_pipeline.py DEVICEIDL DEVICEIDR
    
    # With custom config and duration:
    python src/rcs_specific/delta_power_pipeline.py DEVICEIDL DEVICEIDR \
        --config configs/rcs_configs/baseline_delta_reward.yaml \
        --duration 12.0
"""

import sys
from pathlib import Path
import pickle
import glob
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import duckdb
from typing import List, Dict, Any
from sklearn.linear_model import LinearRegression
import yaml
import argparse
from datetime import timedelta
import logging
import os

# TODO: Double check this file!

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_device_data(device_id: str, config: Dict) -> tuple[pl.DataFrame, duckdb.DuckDBPyConnection]:
    """
    Execute query for the specified device and return results as a Polars DataFrame.
    
    Parameters
    ----------
    device_id : str
        Device ID to query data for
    config : Dict
        Configuration dictionary containing database settings
        
    Returns
    -------
    tuple[pl.DataFrame, duckdb.DuckDBPyConnection]
        Query results as a Polars DataFrame and database connection
    """
    con = duckdb.connect(config["database_path"], read_only=True)
    query = config["query"].format(device=device_id)
    return con.sql(query).pl(), con

def get_session_settings(device_id: str, session: str, config: Dict) -> pl.DataFrame:
    """
    Get session settings from CSV file.
    
    Parameters
    ----------
    device_id : str
        Device ID
    session : str
        Session ID
    config : Dict
        Configuration dictionary containing session settings path template
        
    Returns
    -------
    pl.DataFrame
        Session settings as a Polars DataFrame
    """
    settings_path = config["session_settings_csv_template_path"].format(
        key=device_id, 
        session=session
    )
    return pl.read_csv(settings_path)

def verify_session_settings(session_settings: dict, settings_qa_dict: dict) -> bool:
    """
    Verify that session settings match expected values.
    
    Parameters
    ----------
    session_settings : dict
        Session settings dictionary
    settings_qa_dict : dict
        Expected settings dictionary
        
    Returns
    -------
    bool
        True if settings match, False otherwise
    """
    for key, value in settings_qa_dict.items():
        if key not in session_settings.keys():
            print(f"{key} not found in session_settings_df")
            return False
        if session_settings[key] != value:
            print(f"{key} does not match expected value")
            print(f"Expected: {value}, Actual: {session_settings[key]}")
            return False
    return True

def average_over_time_segments(
    df: pl.DataFrame,
    time_column: str,
    columns_to_average: List[str],
    interval: str = "15s",
    period: str = "15s",
    group_by: List[str] = []
) -> pl.DataFrame:
    """
    Average over time segments.
    
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to average
    time_column : str
        Column containing timestamps
    columns_to_average : List[str]
        Columns to average
    interval : str
        Interval for grouping
    period : str
        Period for grouping
    group_by : List[str], optional
        Additional columns to group by
        
    Returns
    -------
    pl.DataFrame
        Averaged DataFrame
    """
    cols = cs.by_name(*columns_to_average)
    return (
        df.sort(time_column)
        .group_by_dynamic(
            time_column,
            every=interval,
            period=period,
            group_by=group_by
        )
        .agg(cols.mean())
    )

def get_model(device_id: str, config: Dict) -> Any:
    """
    Load model from pickle file.
    
    Parameters
    ----------
    device_id : str
        Device ID
    config : Dict
        Configuration dictionary containing model path template
        
    Returns
    -------
    Any
        Loaded model
    """
    model_path = config["model_path"].format(device_id=device_id)
    return pickle.load(open(glob.glob(model_path)[0], "rb"))

def add_state_predictions(df: pl.DataFrame, model: Any) -> pl.DataFrame:
    """
    Add state predictions to DataFrame.
    
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to add predictions to
    model : Any
        Model to use for predictions. Assumed to be unsupervised_classification model generated from
        integrated_rcs_analysis repository.
        
    Returns
    -------
    pl.DataFrame
        DataFrame with state predictions
    """
    return (
        df.with_columns([
            pl.Series(
                name='State',
                values=model.classifier_model.predict(
                    df.select('^Power_Band.*[5-8]$').to_numpy()
                )
            )
        ])
        .with_columns([
            pl.when(pl.col('State') == 0)
            .then(pl.lit('State 0'))
            .otherwise(pl.lit('State 1'))
            .alias('Adaptive_CurrentAdaptiveState'),
            pl.lit(1).alias('Adaptive_CurrentProgramAmplitudesInMilliamps_1')
        ])
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process and analyze delta power data from RCS devices")
    parser.add_argument(
        "devices",
        nargs="+",
        help="Device IDs to process (e.g., RCS09L RCS09R)"
    )
    parser.add_argument(
        "--config",
        default="configs/rcs_configs/baseline_delta_reward.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=12.0,
        help="Duration in hours to extrapolate data (default: 12.0)"
    )
    return parser.parse_args()

def extrapolate_session_data(
    session_data: pl.DataFrame,
    duration_hours: float,
    power_col: str = 'Power_Band5'
) -> pl.DataFrame:
    """
    Extrapolate session data using linear regression up to specified duration.
    Also adds cumulative average of power values (cumsum/row_number).
    
    Args:
        session_data: DataFrame containing session data
        duration_hours: Duration to extrapolate to in hours
        power_col: Column name for power values to extrapolate
    
    Returns:
        DataFrame with extrapolated values and cumulative average
    """
    # Calculate time since start for existing data
    first_timestamp = session_data['localTime'].min()
    session_data = session_data.with_columns([
        (pl.col('localTime') - first_timestamp).dt.total_seconds()
        .alias('Time_Since_Start_Seconds')
    ])
    
    # Fit linear regression
    x = session_data['Time_Since_Start_Seconds'].to_numpy().reshape(-1, 1)
    y = session_data[power_col].to_numpy()
    model = LinearRegression()
    model.fit(x, y)
    
    # Create time points for extrapolation
    max_seconds = duration_hours * 3600
    current_max_time = session_data['Time_Since_Start_Seconds'].max()
    if current_max_time < max_seconds:
        # Create new time points
        new_times = np.arange(current_max_time + 15, max_seconds + 15, 15)  # 15-second intervals
        new_x = new_times.reshape(-1, 1)
        new_y = model.predict(new_x)
        
        # Create new timestamps
        new_timestamps = [
            first_timestamp + timedelta(seconds=float(t))
            for t in new_times
        ]
        
        # Create extrapolated dataframe
        extrapolated_df = pl.DataFrame({
            'localTime': new_timestamps,
            'Time_Since_Start_Seconds': new_times,
            power_col: new_y,
            'is_extrapolated': [True] * len(new_times)
        })
        
        # Add extrapolation flag to original data
        session_data = session_data.with_columns(
            pl.lit(False).alias('is_extrapolated')
        )
        
        # Combine original and extrapolated data
        result_df = pl.concat([
            session_data,
            extrapolated_df
        ]).sort('localTime')
        
    else:
        result_df = session_data.with_columns(
            pl.lit(False).alias('is_extrapolated')
        )
    
    # Add cumulative average (cumsum/row_number)
    result_df = result_df.with_columns([
        pl.col(power_col).cum_sum().div(
            pl.arange(1, len(result_df) + 1)
        ).alias(f'{power_col}_cumulative_avg')
    ])
    
    return result_df

def extrapolate_column(
    df: pl.DataFrame, 
    column_name: str = 'Power_Band5',
    extrapolation_duration_hours: float = 2.0
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
    new_times = np.arange(current_max_time + 15, current_max_time + extrapolation_seconds + 15, 15)  # 15-second intervals
    
    # Generate predictions
    new_predictions = model.predict(new_times.reshape(-1, 1))
    
    # Create new timestamps using the same timezone as the original data
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
    })
    
    # Add extrapolation flag to original data
    df = df.with_columns(
        pl.lit(False).alias('is_extrapolated')
    )
    
    # Combine original and extrapolated data
    result_df = pl.concat([
        df,
        extrapolated_df
    ]).sort('localTime')
    
    return result_df

def process_device_data(device_id: str, config: Dict, duration_hours: float) -> tuple:
    """
    Process data for a single device.
    
    Parameters
    ----------
    device_id : str
        Device ID to process
    config : Dict
        Configuration dictionary
    duration_hours : float
        Duration to extrapolate data to in hours
        
    Returns
    -------
    tuple[pl.DataFrame, duckdb.DuckDBPyConnection]
        Processed DataFrame and database connection
        
    Notes
    -----
    The function performs several checks and processing steps:
    1. Verifies session settings match expected values
    2. Checks if recordings are at least 2 hours long
    3. Averages data over time segments
    4. Adds state predictions
    5. Extrapolates data to specified duration
    """
    logger = logging.getLogger(f"delta_power_{device_id}")
    
    # Get device data and database connection
    logger.info("Connecting to database and getting device data")
    df, con = get_device_data(device_id, config)
    
    # Filter sessions based on settings and duration
    logger.info("Checking session settings and durations")
    bad_sessions = []
    min_duration_hours = 2  # Minimum required duration in hours
    
    for session in df.select('SessionNumber').unique().to_series().to_list():
        # Check session settings
        session_settings = get_session_settings(device_id, session, config)
        if not verify_session_settings(
            session_settings.to_dict(as_series=False),
            config["settings_qa_dict"]
        ):
            logger.warning(f"Session {session} has incorrect settings")
            bad_sessions.append(session)
            continue
        
        # Check session duration
        session_data = df.filter(pl.col('SessionNumber') == session)
        sample_rate = config["settings_qa_dict"]["TDsampleRates"][0]  # Get sample rate from config
        MIN_DURATION_HOURS = 2  # Minimum required duration in hours
        SECONDS_PER_HOUR = 3600
        min_samples = MIN_DURATION_HOURS * SECONDS_PER_HOUR * sample_rate  # Convert hours to samples
        
        if len(session_data) < min_samples:
            duration_hours = len(session_data) / (sample_rate * 3600)
            logger.warning(
                f"Session {session} is too short: {duration_hours:.2f} hours "
                f"(minimum required: {min_duration_hours} hours)"
            )
            bad_sessions.append(session)
    
    if bad_sessions:
        logger.warning(
            f"Removing {len(bad_sessions)} sessions due to incorrect settings or insufficient duration: "
            f"{bad_sessions}"
        )
        df = df.filter(~pl.col('SessionNumber').is_in(bad_sessions))
        
        if len(df) == 0:
            logger.error("No valid sessions remaining after filtering")
            raise ValueError("No valid sessions remaining after filtering")
    
    # Average over time segments
    logger.info("Averaging over time segments")
    df_averaged = average_over_time_segments(
        df=df,
        time_column='localTime',
        columns_to_average=['^Power_Band.*[5-8]$'],
        group_by=['SessionNumber']
    )
    
    # Add state predictions
    logger.info("Adding state predictions")
    model = get_model(device_id, config)
    df_averaged = add_state_predictions(df_averaged, model)
    
    # Add extrapolation for each session
    logger.info("Processing and extrapolating each session")
    sessions = df_averaged['SessionNumber'].unique()
    extrapolated_dfs = []
    
    for session in sessions:
        logger.info(f"Processing session {session}")
        session_data = df_averaged.filter(pl.col('SessionNumber') == session)
        extrapolated_df = extrapolate_session_data(
            session_data,
            duration_hours,
            config['delta_power_column']
        )
        extrapolated_df = extrapolated_df.with_columns(
            pl.lit(session).alias('SessionNumber')
        )
        extrapolated_dfs.append(extrapolated_df)
    
    df_averaged = pl.concat(extrapolated_dfs)
    
    # Save each session's data
    logger.info("Saving session data")
    output_dir = Path(config["log_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for session in sessions:
        session_df = df_averaged.filter(pl.col('SessionNumber') == session)
        output_path = output_dir / f"{device_id}/{session}_extrapolated.parquet"
        session_df.write_parquet(output_path)
        logger.info(f"Saved session data to {output_path}")
    
    return df_averaged, con

def create_analysis_plots(df_averaged: pl.DataFrame, device_id: str, config: Dict) -> None:
    """Create and save analysis plots."""
    unique_sessions = df_averaged['SessionNumber'].unique().to_list()
    num_sessions = len(unique_sessions)
    
    # Create figure
    fig, axes = plt.subplots(
        num_sessions + 1, 2,
        figsize=(16, 3*(num_sessions + 1)),
        gridspec_kw={'width_ratios': [3, 1]}
    )
    
    # Calculate global max time
    max_time = 0
    for session in unique_sessions:
        session_data = df_averaged.filter(pl.col('SessionNumber') == session)
        if len(session_data) > 0:
            first_timestamp = session_data['localTime'].min()
            session_data = session_data.with_columns(
                (pl.col('localTime') - first_timestamp).dt.total_seconds()
                .alias('Time_Since_Start_Seconds')
            )
            max_time = max(max_time, session_data['Time_Since_Start_Seconds'].max())
    
    max_time_with_extension = max_time * 1.2
    all_sessions_data = []
    all_regression_lines = []
    
    # Plot each session
    for i, session in enumerate(unique_sessions):
        session_data = df_averaged.filter(
            (pl.col('SessionNumber') == session) &
            (pl.col('State') == 1)
        )[1:]
        
        all_session_data = df_averaged.filter(pl.col('SessionNumber') == session)
        
        if len(session_data) > 0:
            first_timestamp = all_session_data['localTime'].min()
            session_data = session_data.with_columns(
                (pl.col('localTime') - first_timestamp).dt.total_seconds()
                .alias('Time_Since_Start_Seconds')
            )
            
            session_pd = session_data.to_pandas()
            avg_delta_power = session_pd[config["delta_power_column"]].mean()
            
            # Plot delta power
            axes[i, 0].plot(
                session_pd['Time_Since_Start_Seconds'],
                session_pd[config["delta_power_column"]],
                label=f"Session {session[-4:]}",
                color='blue'
            )
            
            # Add cumulative average line
            axes[i, 0].plot(
                session_pd['Time_Since_Start_Seconds'],
                session_pd[f"{config['delta_power_column']}_cumulative_avg"],
                label="Cumulative Average",
                color='purple',
                linestyle=':'
            )
            
            all_sessions_data.append({
                'session_id': session[-4:],
                'times': session_pd['Time_Since_Start_Seconds'].values,
                'delta_power': session_pd[config["delta_power_column"]].values,
                'avg_delta_power': avg_delta_power,
                'cumulative_avg': session_pd[f"{config['delta_power_column']}_cumulative_avg"].values
            })
            
            # Add average line
            axes[i, 0].axhline(
                y=avg_delta_power,
                color='green',
                linestyle='-',
                label=f'Overall Avg: {avg_delta_power:.2f}'
            )
            
            # Fit regression line
            if len(session_pd) > 1:
                x = session_pd['Time_Since_Start_Seconds'].values.reshape(-1, 1)
                y = session_pd[config["delta_power_column"]].values
                model = LinearRegression()
                model.fit(x, y)
                
                x_line = np.array([0, max_time_with_extension])
                y_line = model.predict(x_line.reshape(-1, 1))
                
                all_regression_lines.append({
                    'session_id': session[-4:],
                    'x_line': x_line,
                    'y_line': y_line,
                    'slope': model.coef_[0]
                })
                
                last_x = max(x)[0]
                last_y_pred = model.predict(np.array([[last_x]]))[0]
                
                axes[i, 0].plot(
                    x_line,
                    y_line,
                    color='red',
                    linestyle='--',
                    label=f'Slope: {model.coef_[0]:.2f}, End: {last_y_pred:.2f}'
                )
                axes[i, 0].scatter(last_x, last_y_pred, color='red', s=50, zorder=5)
            
            axes[i, 0].legend()
            axes[i, 0].set_ylabel('Delta Power')
            axes[i, 0].set_title(f"Session {session[-4:]}", loc='right')
            axes[i, 0].grid(True, linestyle='--', alpha=0.4)
            axes[i, 0].set_xlim(0, max_time_with_extension)
            
            # Plot time deltas
            if len(all_session_data) > 1:
                all_session_data = all_session_data.sort('localTime')
                time_deltas = all_session_data.with_columns(
                    pl.col('localTime').diff().dt.total_seconds()
                    .alias('time_delta_seconds')
                ).filter(pl.col('time_delta_seconds').is_not_null())
                
                if len(time_deltas) > 0:
                    time_deltas_pd = time_deltas.select('time_delta_seconds').to_pandas()
                    axes[i, 1].hist(
                        time_deltas_pd['time_delta_seconds'],
                        bins=20,
                        color='green',
                        alpha=0.7,
                        log=True
                    )
                    axes[i, 1].set_title('Time Deltas Distribution (Log Scale)')
                    axes[i, 1].set_xlabel('Time Delta (seconds)')
                    axes[i, 1].set_ylabel('Log Frequency')
    
    # Create overlay plot
    overlay_ax = axes[num_sessions, 0]
    overlay_ax.set_title("All Sessions Overlay", fontsize=14)
    overlay_ax.set_xlabel('Time Since Recording Start (seconds)')
    overlay_ax.set_ylabel('Delta Power')
    overlay_ax.grid(True, linestyle='--', alpha=0.4)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_sessions_data)))
    for idx, session_data in enumerate(all_sessions_data):
        overlay_ax.plot(
            session_data['times'],
            session_data['delta_power'],
            label=f"Session {session_data['session_id']}",
            color=colors[idx],
            alpha=0.5
        )
        # Add cumulative average to overlay plot
        overlay_ax.plot(
            session_data['times'],
            session_data['cumulative_avg'],
            label=f"Cumulative Avg {session_data['session_id']}",
            color=colors[idx],
            linestyle=':',
            alpha=0.7
        )
        overlay_ax.axhline(
            y=session_data['avg_delta_power'],
            color=colors[idx],
            linestyle='-',
            alpha=0.3
        )
    
    for idx, reg_line in enumerate(all_regression_lines):
        overlay_ax.plot(
            reg_line['x_line'],
            reg_line['y_line'],
            color=colors[idx],
            linestyle='--',
            alpha=0.7,
            label=f"Trend {reg_line['session_id']} (slope: {reg_line['slope']:.2f})"
        )
    
    overlay_ax.set_xlim(0, max_time_with_extension)
    overlay_ax.legend(loc='upper right', fontsize='small')
    
    # Final plot settings
    axes[num_sessions, 1].axis('off')
    for ax in axes[:-1, 0]:
        ax.set_xlabel('Time Since Recording Start (seconds)')
    
    fig.suptitle(
        f'Delta Power in State 1 and Time Delta Distributions by Session - {device_id}',
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    output_dir = Path(config["log_dir"].format(participant=device_id[:5]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"delta_power_analysis_{device_id}.png"
    plt.savefig(output_path)
    plt.close()

def setup_logging(log_dir: str, device_id: str) -> logging.Logger:
    """
    Set up logging for the pipeline.
    
    Parameters
    ----------
    log_dir : str
        Directory to save log files
    device_id : str
        Device ID for the log file name
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    # Format log_dir if it contains {participant}
    if '{participant}' in log_dir:
        log_dir = log_dir.format(participant=device_id[:-1])
    
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{device_id}_delta_power_analysis.log")
    
    logger = logging.getLogger(f"delta_power_{device_id}")
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def main():
    """
    Main function to run the pipeline.
    
    Processes each device specified in command line arguments:
    1. Sets up logging
    2. Loads and formats configuration
    3. Processes device data
    4. Creates analysis plots
    5. Handles any errors that occur during processing
    """
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)

    # Add the model module path so that the model file can be unpickled. It's a custom class...
    sys.path.append(config["model_module_path"])
    
    for device_id in args.devices:
        # Setup logging
        logger = setup_logging(config["log_dir"], device_id)
        logger.info(f"Processing device: {device_id}")
        
        try:
            # Format any paths in config that contain {participant}
            formatted_config = config.copy()
            for key, value in config.items():
                if isinstance(value, str) and '{participant}' in value:
                    formatted_config[key] = value.format(participant=device_id[:-1])
            
            logger.info("Getting device data and processing")
            df_averaged, con = process_device_data(device_id, formatted_config, args.duration)
            
            logger.info("Creating analysis plots")
            create_analysis_plots(df_averaged, device_id, formatted_config)
            
            con.close()
            logger.info(f"Successfully processed {device_id}")
            
        except Exception as e:
            logger.error(f"Error processing {device_id}: {e}")
            logger.exception("Exception details:")

def plot_nights_power_band(nights: list[pl.DataFrame], power_column: str = 'Power_Band5') -> None:
    """
    Plot specified power band for each DataFrame in nights list, with time since recording start on x-axis.
    
    Parameters
    ----------
    nights : list[pl.DataFrame]
        List of DataFrames containing power band and localTime columns
    power_column : str, optional
        Name of the power band column to plot, by default 'Power_Band5'
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure with one subplot per night
    num_nights = len(nights)
    fig, axes = plt.subplots(num_nights, 1, figsize=(12, 3*num_nights))
    if num_nights == 1:
        axes = [axes]  # Ensure axes is always a list
    
    # Find global max time for consistent x-axis
    max_time = 0
    for df in nights:
        if len(df) > 0:
            first_timestamp = df['localTime'].min()
            df = df.with_columns(
                (pl.col('localTime') - first_timestamp).dt.total_seconds().alias('Time_Since_Start_Seconds')
            )
            max_time = max(max_time, df['Time_Since_Start_Seconds'].max())
    
    # Add 20% to max_time for consistent extension
    max_time_with_extension = max_time * 1.2
    
    # Plot each night
    for i, df in enumerate(nights):
        if len(df) > 0:
            # Calculate time since start
            first_timestamp = df['localTime'].min()
            df = df.with_columns(
                (pl.col('localTime') - first_timestamp).dt.total_seconds().alias('Time_Since_Start_Seconds')
            )
            
            # Convert to pandas for easier plotting
            df_pd = df.to_pandas()
            
            # Plot power band
            axes[i].plot(
                df_pd['Time_Since_Start_Seconds'],
                df_pd[power_column],
                label=power_column,
                color='blue'
            )
            
            # Add average line
            avg_power = df_pd[power_column].mean()
            axes[i].axhline(
                y=avg_power,
                color='green',
                linestyle='-',
                label=f'Avg: {avg_power:.2f}'
            )
            
            # Add regression line if enough points
            if len(df_pd) > 1:
                x = df_pd['Time_Since_Start_Seconds'].values.reshape(-1, 1)
                y = df_pd[power_column].values
                model = LinearRegression()
                model.fit(x, y)
                
                x_line = np.array([0, max_time_with_extension])
                y_line = model.predict(x_line.reshape(-1, 1))
                
                axes[i].plot(
                    x_line,
                    y_line,
                    color='red',
                    linestyle='--',
                    label=f'Slope: {model.coef_[0]:.2f}'
                )
            
            # Set plot properties
            axes[i].set_title(f'Night {i+1}')
            axes[i].set_ylabel(power_column)
            axes[i].grid(True, linestyle='--', alpha=0.4)
            axes[i].set_xlim(0, max_time_with_extension)
            axes[i].legend()
    
    # Set common x-axis label
    for ax in axes:
        ax.set_xlabel('Time Since Recording Start (seconds)')
    
    # Add overall title
    fig.suptitle(f'{power_column} Over Time for Each Night', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

if __name__ == "__main__":
    main() 