import polars as pl
import polars.selectors as cs
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import pickle
import glob


def get_device_data(session_data_template_path, device_id, session_type, sessions_group, sort_col='localTime', non_null_col='Power_Band5', first_n_of_each_session_to_drop=0):
    # Collect datat for current session_group
    # Non-null col corresponds to column in which we drop null rows
    df = pl.read_parquet(session_data_template_path.format(session_type=session_type, device=device_id, sessions=sessions_group)).filter(pl.col('Power_Band5').is_not_null())


    # Convert the first occurrence of Power_Band5 to null for each new SessionNumber
    if first_n_of_each_session_to_drop > 0:
        df = df.with_columns(
            pl.when(pl.col("SessionNumber") != pl.col("SessionNumber").shift(first_n_of_each_session_to_drop, fill_value='dummy'))
            .then(pl.lit(None))
            .otherwise(pl.lit(1))
            .alias("First_Sample_Indicator")
        ).filter(pl.col('First_Sample_Indicator').is_not_null()).drop('First_Sample_Indicator')
        
    return df.sort(pl.col(sort_col))


def get_session_settings(device_id, session, sesison_type, session_settings_csv_template_path):
    session_settings_csv_path = session_settings_csv_template_path.format(device=device_id, session=session, session_type=sesison_type)
    session_settings_df = pl.read_csv(session_settings_csv_path)
    return session_settings_df


def verify_session_settings(session_settings: dict, settings_QA_dict: dict):
    for key, value in settings_QA_dict.items():
        if key not in session_settings.keys():
            print(f"{key} not found in session_settings_df")
            return False
        if session_settings[key] != value:
            if len(session_settings[key]) > 1 and all([v == value[0] for v in session_settings[key]]):
                continue
            print(f"{key} does not match expected value")
            print(f"Expected: {value}, Actual: {session_settings[key]}")
            return False
    return True


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


def average_over_time_segments_ensure_consistent_size(df: pl.DataFrame, time_column: str, columns_to_average: List[str], interval: str, period: str, group_by: List[str] = [], filter_col = 'Power_Band5') -> pl.DataFrame:
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
            cols
        ).filter(
            pl.col(filter_col).list.len() == pl.col(filter_col).list.len().max()
        ).select(
            pl.col(group_by).first(),
            pl.col(time_column),
            cols.list.mean()
        )
    )
    return df_averaged


def get_model(device_id, model_path, model_group):
    return pickle.load(open(glob.glob(model_path.format(device_id=device_id, model_group=model_group))[0], "rb"))


def add_state_predictions(df, linear_model, feature_columns=['^Power_Band.*[5-8]$']):
    return df.with_columns(
        pl.Series(name='State', values=linear_model.classifier_model.predict(df.select(feature_columns).to_numpy()))
    ).with_columns(
        pl.when(pl.col('State') == 0).then(pl.lit('Wake+REM')).otherwise(pl.lit('NREM')).alias('Adaptive_CurrentAdaptiveState_mapped'),
        # pl.lit(1).alias('Adaptive_CurrentProgramAmplitudesInMilliamps_1') # Give dummy amplitude to calculate reward
    )



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
    extrapolation_seconds = extrapolation_duration_hours * 3600 # 3600 = number of seconds in an hour
    new_times = np.arange(current_max_time + num_seconds_for_power_averaging, current_max_time + extrapolation_seconds + num_seconds_for_power_averaging, num_seconds_for_power_averaging)  # 15-second intervals
    
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
    }).with_columns(pl.col(column_name).clip(0, None))
    
    # Add extrapolation flag to original data
    df = df.with_columns(
        pl.lit(False).alias('is_extrapolated')
    )
    
    # Combine original and extrapolated data
    result_df = pl.concat([
            df,
            extrapolated_df,
        ],
        how='diagonal_relaxed'
    ).sort('localTime')
    
    return result_df


def add_running_average(df: pl.DataFrame, column_name: str) -> pl.DataFrame:
    return df.with_row_index(offset=1).with_columns(
        (pl.col(column_name).cum_sum() / pl.col('index')).alias(f'{column_name}_running_average')
    ).drop('index')

def plot_nights_power_band(
    nights: list[pl.DataFrame], 
    power_column: str = 'Power_Band5', 
    running_average_column: str = 'Power_Band5_running_average', 
    title_col: str = 'SessionNumber',
    max_x_axis: int = 40_000
) -> plt.Figure:
    """
    Plot specified power band for each DataFrame in nights list, with time since recording start on x-axis.
    
    Parameters
    ----------
    nights : list[pl.DataFrame]
        List of DataFrames containing power band and localTime columns
    power_column : str, optional
        Name of the power band column to plot, by default 'Power_Band5'
    running_average_column : str, optional
        Name of the running average column to plot, by default 'Power_Band5_running_average'
    title_col : str, optional
        Name of the column to use for subplot titles, by default 'SessionNumber'
    
    Returns
    -------
    plt.Figure
        The figure object that can be saved to a file
    """
    
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
    
    # Plot each night
    for i, df in enumerate(nights):
        if len(df) > 0:
            # Calculate time since start
            first_timestamp = df['localTime'].min()
            df = df.with_columns(
                (pl.col('localTime') - first_timestamp).dt.total_seconds().alias('Time_Since_Start_Seconds')
            )
            
            # Convert to pandas for easier plotting
            title_value = df.select(title_col).filter(pl.col(title_col).is_not_null()).unique().get_column(title_col).to_list()
            df_pd = df.to_pandas()
            
            # Get title from specified column
            # title_value = "Night " + str(i+1)
            # print(df[title_col])
            # if title_col in df.columns and df[title_col].n_unique() == 1:
            #     title_value = str(df[title_col].unique()[0])
            
            # Plot power band
            axes[i].plot(
                df_pd['Time_Since_Start_Seconds'],
                df_pd[power_column],
                label=power_column,
                color='blue'
            )
            
            # Plot running average
            axes[i].plot(
                df_pd['Time_Since_Start_Seconds'],
                df_pd[running_average_column],
                label=running_average_column,
                color='purple',
                linewidth=2
            )
            
            # Add average line - only using non-extrapolated data
            non_extrapolated_df = df_pd[df_pd['is_extrapolated'] == False]
            avg_power = non_extrapolated_df[power_column].mean()
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
                
                x_line = np.array([0, max_x_axis])
                y_line = model.predict(x_line.reshape(-1, 1))
                
                axes[i].plot(
                    x_line,
                    y_line,
                    color='red',
                    linestyle='--',
                    label=f'Slope: {model.coef_[0]:.2f}'
                )
            
            # Set plot properties
            axes[i].set_title(title_value)
            axes[i].set_ylabel(power_column)
            axes[i].grid(True, linestyle='--', alpha=0.4)
            axes[i].set_xlim(0, max_x_axis)
            axes[i].legend()
    
    # Set common x-axis label
    for ax in axes:
        ax.set_xlabel('Time Since Recording Start (seconds)')
    
    # Add overall title
    fig.suptitle(f'{power_column} Over Time for Each Night', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig


def plot_nights_state_predictions(
    nights: list[pl.DataFrame], 
    state_column: str = 'Adaptive_CurrentAdaptiveState_mapped', 
    title_col: str = 'SessionNumber',
    max_x_axis: int = 40_000
) -> plt.Figure:
    """
    Plot state predictions for each DataFrame in nights list, with time since recording start on x-axis.
    
    Parameters
    ----------
    nights : list[pl.DataFrame]
        List of DataFrames containing state predictions and localTime columns
    state_column : str, optional
        Name of the state column to plot, by default 'Adaptive_CurrentAdaptiveState_mapped'
    title_col : str, optional
        Name of the column to use for subplot titles, by default 'SessionNumber'
    max_x_axis : int, optional
        Maximum x-axis value in seconds, by default 40_000
    
    Returns
    -------
    plt.Figure
        The figure object that can be saved to a file
    """
    
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
    
    # Plot each night
    for i, df in enumerate(nights):
        if len(df) > 0:
            # Calculate time since start
            first_timestamp = df['localTime'].min()
            df = df.with_columns(
                (pl.col('localTime') - first_timestamp).dt.total_seconds().alias('Time_Since_Start_Seconds')
            )
            
            # Convert to pandas for easier plotting
            title_value = df.select(title_col).filter(pl.col(title_col).is_not_null()).unique().get_column(title_col).to_list()
            df_pd = df.to_pandas()
            
            # Create state mapping for plotting
            state_mapping = {'Wake+REM': 0, 'NREM': 1}
            df_pd['State_Numeric'] = df_pd[state_column].map(state_mapping)
            
            # Plot state predictions as a step function
            axes[i].step(
                df_pd['Time_Since_Start_Seconds'],
                df_pd['State_Numeric'],
                where='post',
                label=state_column,
                color='blue',
                linewidth=2
            )
            
            # Add horizontal lines for each state
            axes[i].axhline(y=0, color='lightblue', linestyle='-', alpha=0.3, label='Wake+REM')
            axes[i].axhline(y=1, color='darkblue', linestyle='-', alpha=0.3, label='NREM')
            
            # Calculate state statistics
            state_counts = df_pd[state_column].value_counts()
            total_samples = len(df_pd)
            
            # Add text box with state statistics
            stats_text = f"Wake+REM: {state_counts.get('Wake+REM', 0)} ({state_counts.get('Wake+REM', 0)/total_samples*100:.1f}%)\n"
            stats_text += f"NREM: {state_counts.get('NREM', 0)} ({state_counts.get('NREM', 0)/total_samples*100:.1f}%)"
            
            axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Set plot properties
            axes[i].set_title(title_value)
            axes[i].set_ylabel('Sleep State')
            axes[i].set_ylim(-0.1, 1.1)
            axes[i].set_yticks([0, 1])
            axes[i].set_yticklabels(['Wake+REM', 'NREM'])
            axes[i].grid(True, linestyle='--', alpha=0.4)
            axes[i].set_xlim(0, max_x_axis)
            axes[i].legend(loc='upper right')
    
    # Set common x-axis label
    for ax in axes:
        ax.set_xlabel('Time Since Recording Start (seconds)')
    
    # Add overall title
    fig.suptitle(f'State Predictions Over Time for Each Night', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig

def find_binary_pattern_match(df: pl.DataFrame, column: str, window: list | np.ndarray, 
                             match_threshold: float = 1.0) -> tuple[int, pl.DataFrame]:
    """
    Optimized version specifically for binary vectors.
    Uses Hamming similarity (fraction of matching bits).
    
    Parameters:
    -----------
    df : pl.DataFrame
        Input dataframe
    column : str
        Column name with binary values
    window : list | np.ndarray
        Binary pattern to match
    match_threshold : float
        Fraction of bits that must match (1.0 = exact match)
    
    Returns:
    --------
    tuple[int, pl.DataFrame] : (first_matching_index, original_dataframe)
    """
    
    window_size = len(window)
    window = np.array(window, dtype=np.int8)
    
    col_values = df.get_column(column).to_numpy(zero_copy_only=False).astype(np.int8)
    
    if len(col_values) < window_size:
        return (-1, df)
    
    # For binary vectors, check fraction of matching elements
    for i in range(len(col_values) - window_size + 1):
        segment = col_values[i:i+window_size]
        
        # Hamming similarity: fraction of matching bits
        matches = np.sum(segment == window) / window_size
        
        if matches >= match_threshold:
            return i