"""
Pipeline to establish participant reward statistics from baseline data.
This script processes RCS data to calculate delta power statistics for Bayesian optimization.
First calculates the mean delta power in state 1 for each session, then averages these values across nights.

Provides z-score metrics for each device, so that new sessions can be compared to the participant's baseline.

Example usage:
    python src/rcs_specific/establish_participant_reward_stats_from_baseline_data.py RCS09L RCS09R
    
    # With custom config:
    python src/rcs_specific/establish_participant_reward_stats_from_baseline_data.py RCS09L --config configs/rcs_configs/baseline_stats_for_each_device_reward.yaml
"""

import os
import pickle
import glob
import argparse
import logging
from typing import List, Dict, Any

import polars as pl
import polars.selectors as cs
import duckdb
import yaml
import matplotlib.pyplot as plt
import numpy as np

import sys

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
    # Check if log_dir contains unfilled template variables
    if '{' in log_dir and '}' in log_dir:
        # Check specifically for {participant} variable
        if '{participant}' in log_dir:
            # Extract participant ID from device_id (assuming format like RCS09L where RCS09 is the participant)
            log_dir = log_dir.replace('{participant}', device_id[:-1])
        
        # Check for any other unfilled variables and log a warning
        import re
        if re.search(r'\{[^}]+\}', log_dir):
            print(f"WARNING: log_dir contains unfilled variables: {log_dir}")
            # Replace any remaining variables with 'unknown'
            log_dir = re.sub(r'\{[^}]+\}', 'unknown', log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{device_id}_reward_stats.log")
    
    logger = logging.getLogger(f"reward_stats_{device_id}")
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
        config = yaml.safe_load(f)
    
    # Add model module path to sys.path if specified in config
    if 'model_module_path' in config:
        sys.path.append(config['model_module_path'])
    
    return config


def get_device_data(device_id: str, con: duckdb.DuckDBPyConnection, query: str) -> pl.DataFrame:
    """
    Execute query for the specified device and return results as a Polars DataFrame.
    
    Parameters
    ----------
    device_id : str
        Device ID to query data for
    con : duckdb.DuckDBPyConnection
        DuckDB connection
    query : str
        SQL query template with {device} placeholder
        
    Returns
    -------
    pl.DataFrame
        Query results as a Polars DataFrame
    """
    query_formatted = query.format(device=device_id)
    df = con.sql(query_formatted).pl()
    return df


def get_session_settings(device_id: str, session: str, session_settings_csv_template_path: str) -> pl.DataFrame:
    """
    Get session settings from CSV file.
    
    Parameters
    ----------
    device_id : str
        Device ID
    session : str
        Session ID
    session_settings_csv_template_path : str
        Template path to session settings CSV file
        
    Returns
    -------
    pl.DataFrame
        Session settings as a Polars DataFrame
    """
    session_settings_csv_path = session_settings_csv_template_path.format(key=device_id, session=session)
    session_settings_df = pl.read_csv(session_settings_csv_path)
    return session_settings_df


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
            return False
        if session_settings[key] != value:
            return False
    return True


def average_over_time_segments(
    df: pl.DataFrame, 
    time_column: str, 
    columns_to_average: List[str], 
    interval: str, 
    period: str, 
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


def get_model(device_id: str, model_path: str) -> Any:
    """
    Load model from pickle file.
    
    Parameters
    ----------
    device_id : str
        Device ID
    model_path : str
        Template path to model file
        
    Returns
    -------
    Any
        Loaded model
    """
    model_glob = model_path.format(device_id=device_id)
    model_files = glob.glob(model_glob)
    if not model_files:
        raise FileNotFoundError(f"No model file found for {device_id} at {model_glob}")
    return pickle.load(open(model_files[0], "rb"))


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
    return df.with_columns(
        pl.Series(name='State', values=model.classifier_model.predict(df.select('^Power_Band.*[5-8]$').to_numpy()))
    )
    # .with_columns(
    #     pl.when(pl.col('State') == 0).then(pl.lit('State 0')).otherwise(pl.lit('State 1')).alias('Adaptive_CurrentAdaptiveState'),
    #     pl.lit(1).alias('Adaptive_CurrentProgramAmplitudesInMilliamps_1')  # Give dummy amplitude to calculate reward
    # )


def save_session_data_and_plot(
    averaged_state1_delta_power_for_each_session: pl.DataFrame,
    log_dir: str,
    logger: logging.Logger,
    device_id: str
) -> None:
    """
    Save session data to CSV and create distribution plot.
    
    Parameters
    ----------
    averaged_state1_delta_power_for_each_session : pl.DataFrame
        DataFrame containing mean delta power in state 1 for each session
    log_dir : str
        Directory to save CSV and plot
    logger : logging.Logger
        Logger instance
    device_id : str
        Device ID for creating the subdirectory
    """
    # Create device-specific subdirectory
    device_subdir = os.path.join(log_dir, f"{device_id}_baseline_stats")
    os.makedirs(device_subdir, exist_ok=True)
    
    # Save the data to CSV in the device-specific subdirectory
    csv_path = os.path.join(device_subdir, "averaged_state1_delta_power.csv")
    averaged_state1_delta_power_for_each_session.write_csv(csv_path)
    logger.info(f"Saved averaged state 1 delta power to {csv_path}")
    
    # Create and save a plot of the distribution
    try:
        # Convert to pandas for easier plotting
        pd_df = averaged_state1_delta_power_for_each_session.to_pandas()
        
        # Plot 1: Simple histogram with normal distribution
        plt.figure(figsize=(10, 6))
        plt.hist(pd_df['mean_delta_power_in_state_1'], bins=10, alpha=0.7, 
                color='skyblue', edgecolor='black')
        
        # Add mean line
        mean_value = pd_df['mean_delta_power_in_state_1'].mean()
        plt.axvline(x=mean_value, color='red', linestyle='--', 
                   label=f'Mean: {mean_value:.2f}')
        
        # Add a normal distribution curve
        mu = mean_value
        sigma = pd_df['mean_delta_power_in_state_1'].std()
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        
        # Scale the normal distribution to match histogram height
        hist_height = len(pd_df) * (pd_df['mean_delta_power_in_state_1'].max() - 
                                   pd_df['mean_delta_power_in_state_1'].min()) / 10
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * 
             np.exp(-0.5 * ((x - mu) / sigma) ** 2)) * hist_height
        
        plt.plot(x, y, 'k--', linewidth=1.5, 
                label=f'Normal Dist: μ={mu:.2f}, σ={sigma:.2f}')
        
        plt.title('Distribution of Mean Delta Power in State 1 Across Nights')
        plt.xlabel('Mean Delta Power')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot to the device-specific subdirectory
        plot_path = os.path.join(device_subdir, "delta_power_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved delta power distribution plot to {plot_path}")
        
        # Plot 2: Weighted distribution with weighted mean and std
        plt.figure(figsize=(10, 6))
        
        # Calculate weighted mean and std
        weighted_mean = np.average(
            pd_df['mean_delta_power_in_state_1'], 
            weights=pd_df['Session_Duration_Minutes_Normalized']
        )
        
        weighted_std = np.sqrt(
            np.average(
                (pd_df['mean_delta_power_in_state_1'] - weighted_mean)**2, 
                weights=pd_df['Session_Duration_Minutes_Normalized']
            )
        )
        
        # Create histogram with weights
        plt.hist(
            pd_df['mean_delta_power_in_state_1'], 
            bins=10, 
            alpha=0.7,
            color='lightgreen', 
            edgecolor='black',
            weights=pd_df['Session_Duration_Minutes_Normalized'] * (len(pd_df) / pd_df['Session_Duration_Minutes_Normalized'].sum())
        )
        
        # Add weighted mean line
        plt.axvline(x=weighted_mean, color='red', linestyle='--', 
                   label=f'Weighted Mean: {weighted_mean:.2f}')
        
        # Add a normal distribution curve with weighted parameters
        x = np.linspace(weighted_mean - 3*weighted_std, weighted_mean + 3*weighted_std, 100)
        
        # Scale the normal distribution to match histogram height
        hist_height = len(pd_df) * (pd_df['mean_delta_power_in_state_1'].max() - 
                                   pd_df['mean_delta_power_in_state_1'].min()) / 10
        y = ((1 / (np.sqrt(2 * np.pi) * weighted_std)) * 
             np.exp(-0.5 * ((x - weighted_mean) / weighted_std) ** 2)) * hist_height
        
        plt.plot(x, y, 'k--', linewidth=1.5, 
                label=f'Weighted Normal Dist: μ={weighted_mean:.2f}, σ={weighted_std:.2f}')
        
        plt.title('Weighted Distribution of Mean Delta Power in State 1 Across Nights')
        plt.xlabel('Mean Delta Power')
        plt.ylabel('Weighted Frequency')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the weighted plot
        weighted_plot_path = os.path.join(device_subdir, "weighted_delta_power_distribution.png")
        plt.savefig(weighted_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved weighted delta power distribution plot to {weighted_plot_path}")
        
    except Exception as e:
        logger.warning(f"Could not create delta power distribution plots: {str(e)}")


def calculate_delta_power_stats(
    df_averaged: pl.DataFrame, 
    delta_power_column: str,
    logger: logging.Logger,
    device_id: str
) -> Dict[str, float]:
    """
    Calculate delta power statistics across nights.
    
    Parameters
    ----------
    df_averaged : pl.DataFrame
        Averaged DataFrame with state predictions
    delta_power_column : str
        Column containing delta power values
    logger : logging.Logger
        Logger instance
    device_id : str
        Device ID for creating output subdirectory
        
    Returns
    -------
    Dict[str, float]
        Dictionary with mean and standard deviation of delta power
    """
    # Get the date of the session end-time to aggregate sessions by night
    session_duration_df = df_averaged.join(
        df_averaged.group_by('SessionNumber').agg(
            pl.col('localTime').max().dt.date().alias('Session_Date')
        ),
        on='SessionNumber',
        how='left'
    ).group_by('Session_Date').agg(
        # Calculate the duration of the session(s)
        pl.col('SessionNumber').unique().alias('SessionNumber'),
        (pl.col('localTime').max() - pl.col('localTime').min()).dt.total_minutes().alias('Session_Duration_Minutes')
    ).with_columns(
        # Normalize the session duration to sum to 1
        (pl.col('Session_Duration_Minutes') / pl.col('Session_Duration_Minutes').sum()).alias('Session_Duration_Minutes_Normalized')
    )
    
    # Join session duration information to the averaged data
    df_averaged = df_averaged.join(
        session_duration_df.explode('SessionNumber'), 
        on='SessionNumber', 
        how='left'
    )
    
    # Get the mean delta power in state 1 for each session
    averaged_state1_delta_power_for_each_session = df_averaged.filter(
        pl.col('State') == 1
    ).group_by('Session_Date').agg(
        pl.col(delta_power_column).mean().alias('mean_delta_power_in_state_1'),
        pl.col('Session_Duration_Minutes_Normalized').first()
    )

    # Log the data for each session
    logger.info(f"Mean delta power in state 1 for each session: {averaged_state1_delta_power_for_each_session}")
    
    # Save the data to CSV and create plot
    log_dir = os.path.dirname(logger.handlers[0].baseFilename)
    save_session_data_and_plot(averaged_state1_delta_power_for_each_session, log_dir, logger, device_id)
    
    # Calculate weighted mean and standard deviation. Weighted by session duration.
    stats = averaged_state1_delta_power_for_each_session.select(
        (pl.col('Session_Duration_Minutes_Normalized') * pl.col('mean_delta_power_in_state_1')).sum().alias('delta_mean_across_nights'),
        (
            (
                (
                    pl.col('mean_delta_power_in_state_1') - 
                    (pl.col('Session_Duration_Minutes_Normalized') * pl.col('mean_delta_power_in_state_1')).sum()
                ) ** 2 * 
                pl.col('Session_Duration_Minutes_Normalized')
            ).sum() ** 0.5
        ).alias('delta_std_across_nights')
    )
    
    # Convert to dictionary
    stats_dict = stats.to_dicts()[0]
    return stats_dict


def update_stats_csv(
    device_id: str, 
    stats: Dict[str, float], 
    output_csv_path: str
) -> None:
    """
    Update statistics CSV file with results for the current device.
    
    Parameters
    ----------
    device_id : str
        Device ID
    stats : Dict[str, float]
        Statistics dictionary
    output_csv_path : str
        Path to output CSV file
    """
    # Create a new row for the current device
    new_row = {
        'device_id': device_id,
        'delta_mean': stats['delta_mean_across_nights'],
        'delta_std': stats['delta_std_across_nights']
    }
    
    # Check if the file exists
    if os.path.exists(output_csv_path):
        # Read existing CSV
        df = pl.read_csv(output_csv_path)
        
        # Check if device already exists
        if device_id in df['device_id'].to_list():
            # Update existing row
            df = df.filter(pl.col('device_id') != device_id)
        
        # Append new row
        new_df = pl.concat([df, pl.DataFrame([new_row])])
    else:
        # Create new DataFrame
        new_df = pl.DataFrame([new_row])
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Write to CSV
    new_df.write_csv(output_csv_path)


def process_device(
    device_id: str, 
    config: Dict[str, Any], 
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Process a single device.
    
    Parameters
    ----------
    device_id : str
        Device ID to process
    config : Dict[str, Any]
        Configuration dictionary
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    Dict[str, float]
        Statistics dictionary
    """
    logger.info(f"Processing device: {device_id}")
    
    # Connect to the database
    logger.info(f"Connecting to database: {config['database_path']}")
    con = duckdb.connect(config['database_path'], read_only=True)
    
    # Get device data
    logger.info("Querying device data")
    df = get_device_data(device_id, con, config['query'])
    logger.info(f"Retrieved {df.height} rows of data")
    
    # Check session settings
    logger.info("Checking session settings")
    bad_sessions = []
    good_sessions = []
    
    for session in df.select('SessionNumber').unique().to_series().to_list():
        logger.info(f"Checking session {session}")
        try:
            session_settings = get_session_settings(
                device_id, 
                session, 
                config['session_settings_csv_template_path']
            )
            if not verify_session_settings(
                session_settings.to_dict(as_series=False), 
                config['settings_qa_dict']
            ):
                logger.warning(f"Session {session} has incorrect settings")
                bad_sessions.append(session)
            else:
                logger.info(f"Session {session} has correct settings")
                good_sessions.append(session)
        except Exception as e:
            logger.error(f"Error checking session {session}: {str(e)}")
            bad_sessions.append(session)
    
    if bad_sessions:
        logger.warning(f"Sessions with incorrect settings: {bad_sessions}")
        logger.info("Removing these sessions from the analysis")
        df = df.filter(~pl.col('SessionNumber').is_in(bad_sessions))
    else:
        logger.info("All sessions have correct settings")
    
    # Average over time segments
    logger.info("Averaging over time segments")
    df_averaged = average_over_time_segments(
        df=df,
        time_column='localTime',
        columns_to_average=['^Power_Band.*[5-8]$'],
        interval='15s',
        period='15s',
        group_by=['SessionNumber']
    )
    
    # Get model and add state predictions
    logger.info("Loading model and adding state predictions")
    try:
        model = get_model(device_id, config['model_path'])
        df_averaged = add_state_predictions(df_averaged, model)
    except Exception as e:
        logger.error(f"Error loading model or adding predictions: {str(e)}")
        raise
    
    # Calculate delta power statistics
    logger.info("Calculating delta power statistics")
    stats = calculate_delta_power_stats(df_averaged, config['delta_power_column'], logger, device_id)
    logger.info(f"Delta power statistics: {stats}")
    
    return stats


def main():
    """Main function to run the pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate reward statistics from baseline data")
    parser.add_argument("devices", nargs="+", help="Device IDs to process")
    parser.add_argument("--config", default="configs/reward_stats_config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Process each device
    for device_id in args.devices:
        # Set up logging
        logger = setup_logging(config['log_dir'], device_id)
        
        try:
            # Process device
            stats = process_device(device_id, config, logger)
            
            # Update statistics CSV
            update_stats_csv(device_id, stats, config['output_csv_path'])
            
            logger.info(f"Successfully processed device: {device_id}")
        except Exception as e:
            logger.error(f"Error processing device {device_id}: {str(e)}")
            logger.exception("Exception details:")


if __name__ == "__main__":
    main()
