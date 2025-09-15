import polars as pl
from typing import Dict, Tuple
import json
from pathlib import Path
import bidict
import numpy as np
from prefect import task


def find_binary_pattern_match(df: pl.DataFrame, column: str = None, value_to_match: str = None, window_length: int = None, 
                             match_threshold: float = 1.0) -> tuple[int, pl.DataFrame]:
    """
    Optimized version specifically for binary vectors that finds the first occurrence of a pattern match.
    Uses Hamming similarity (fraction of matching bits) to identify matches.
    
    The function works by:
    1. Converting a string column to binary values (1 for value_to_match, 0 for others)
    2. Creating a window of ones of specified length to match against
    3. Sliding this window along the binary values one position at a time
    4. For each position, calculating what fraction of bits match between the window and data
    5. Returning the first position where the match fraction exceeds the threshold

    The value to match is device determined NREM state. So this will find the first occurence where NREM states exceeds a threshold, given a specific window length and correlation threshold.
    
    Parameters:
    -----------
    df : pl.DataFrame
        Input dataframe
    column : str
        Column name with string values that will be converted to binary
    value_to_match : str
        String value to match (converted to 1, all others to 0)
    window_length : int
        Length of the window pattern to match
    match_threshold : float
        Fraction of bits that must match (1.0 = exact match)
    
    Returns:
    --------
    tuple[int, pl.DataFrame] : (first_matching_index, original_dataframe)
        first_matching_index: Index of first match found, or -1 if no match
        original_dataframe: The input dataframe unchanged
    """
    
    window = np.ones(window_length)

    
    # Convert string column to binary: value_to_match becomes 1, all others become 0
    col_values = df.get_column(column).to_numpy(zero_copy_only=False)
    binary_values = (col_values == value_to_match).astype(np.int8)
    
    if len(binary_values) < window_length:
        return (-1, df)
    
    # For binary vectors, check fraction of matching elements
    for i in range(len(binary_values) - window_length + 1):
        segment = binary_values[i:i+window_length]
        
        # Hamming similarity: fraction of matching bits
        matches = np.sum(segment == window) / window_length
        
        if matches >= match_threshold:
            return i
    
    return -1

def add_running_average(df: pl.DataFrame, column_name: str) -> pl.DataFrame:
    return df.with_row_index(offset=1).with_columns(
        (pl.col(column_name).cum_sum() / pl.col('index')).alias(f'{column_name}_running_average')
    ).drop('index')

@task
def calculate_bilateral_reward_and_sem(
    session_data_dict: Dict[str, pl.DataFrame],
    participant: str,
    nrem_state_base_path: str,
    current_target_amps_path: str,
    baseline_running_stats_path: str,
    baseline_rec_lengths_stats_path: str,
    pattern_match_config_path: str, # This is used to find the first NREM state in the data.
    SECONDS_PER_SAMPLE: float
) -> Dict[str, float]:
    """
    Calculate bilateral reward based on time-weighted average of delta power in NREM state.
    
    The reward is calculated as:
    Reward = (T₁,L/(T₁,L + T₁,R)) * δ₁,L + (T₁,R/(T₁,L + T₁,R)) * δ₁,R
    
    where:
    - T₁,i = time spent in state 1 (putative NREM) for hemisphere i ∈ {L,R}
    - δ₁,i = average delta power in state 1 for hemisphere i ∈ {L,R}
    
    Parameters
    ----------
    session_data_dict : Dict[str, pl.DataFrame]
        Dictionary containing session data for left and right hemispheres
    participant : str
        Participant ID
    Returns
    -------
    float
        Bilateral reward value
    """
    # Load the current target amps from the provided path
    with open(Path(current_target_amps_path.format(
        participant=participant
    )), 'r') as f:
        current_target_amps = json.load(f)

    sides = list(current_target_amps.keys())

    # Calculate time in NREM and average delta power for each hemisphere
    hemisphere_stats = {}
    for side in sides:
        device = participant + side[0] # only take L or R

        # Load the NREM state mapping from the provided path
        with open(Path(nrem_state_base_path.format(
            participant=participant,
            device=device
        )), 'r') as f:
            nrem_state_mapping = json.load(f)

        if "NREM" not in list(nrem_state_mapping.keys()):
            nrem_state_mapping = bidict.bidict(nrem_state_mapping).inv
            assert "NREM" in list(nrem_state_mapping.keys()), "NREM state not found in nrem_state_mapping"

        
        # Load the baseline running stats from the provided path
        baseline_running_stats = pl.read_parquet(baseline_running_stats_path.format(
            participant=participant,
            device=device
        ))

        # Load the pattern match config from the provided path. This is used to find the first NREM state in the data.
        with open(pattern_match_config_path.format(
            participant=participant,
            device=device
        ), 'r') as f:
            pattern_match_config = json.load(f)

        window_length = pattern_match_config["window_length"]
        corr_threshold = pattern_match_config["correlation_threshold"]
        first_n_of_each_session_to_drop_in_fft_intervals = pattern_match_config["first_n_of_each_session_to_drop"]
        
        data = session_data_dict[side]

        # Drop beginning of each session to avoid artifacts.
        if first_n_of_each_session_to_drop_in_fft_intervals > 0:
            # n is in units of FFT intervals. So need get row number that corresponds. 5m samples at 500Hz is the first ~3 minutes of data... probably enough to get past the first artifact.
            row_index_to_slice_at = data.head(5_000_000).with_row_index().filter(pl.col('Power_Band5').is_not_null()).get_column('index').first()
            data = data.slice(row_index_to_slice_at, None)

        # Get data in NREM state
        nrem_delta_power = data.filter(
            # Filter for NREM state
            (pl.col("Adaptive_CurrentAdaptiveState") == nrem_state_mapping["NREM"]) &
            # Filter where the current target amp is at target (to avoid ramping sections)
            (pl.col("Adaptive_CurrentProgramAmplitudesInMilliamps").list.get(0) == current_target_amps[side])
        # Select the columns of interest
        ).select(
            pl.col('localTime'),
            pl.col("Adaptive_Ld0_featureInputs").list.get(0),
            pl.col("Adaptive_CurrentAdaptiveState")
        # Filter out duplicate values of first Ld0. Each FFT cycle prints out identical values until a new updateRate cycle is reached.
        ).filter(
            pl.col("Adaptive_Ld0_featureInputs").diff().abs() != 0
        # Add a column for time since start
        ).with_columns(
            (pl.col("localTime") - pl.col("localTime").min()).dt.total_seconds().cast(pl.Float64).alias("Time_Since_Start_Seconds")
        )

        if nrem_delta_power.height == 0:
            print(f"No NREM data found for {participant} {side}")
            print(f"Target amps: {current_target_amps[side]}... may be incorrect for this Session")
            continue


        inferred_first_n_after_filtering_for_NREM = find_binary_pattern_match(
            nrem_delta_power, 
            'Adaptive_CurrentAdaptiveState', 
            value_to_match=nrem_state_mapping["NREM"], 
            window_length=window_length, 
            match_threshold=corr_threshold
        )
        if inferred_first_n_after_filtering_for_NREM == -1:
            print(f"NREM correlation threshold failed for {participant} {side}...")
            print(f"Target amps: {current_target_amps[side]}... may be incorrect for this Session")
            continue
        else:
            nrem_delta_power = nrem_delta_power.slice(inferred_first_n_after_filtering_for_NREM, None)


        nrem_delta_power = add_running_average(nrem_delta_power, "Adaptive_Ld0_featureInputs")

        # Get baseline mean and std from baseline_running_stats
        baseline_mean_and_std = nrem_delta_power.tail(1).join_asof(
            baseline_running_stats, on='Time_Since_Start_Seconds', strategy='nearest'
        ).select("Adaptive_Ld0_featureInputs_running_average", "Mean", "Std")

        baseline_mean = baseline_mean_and_std.select("Mean").item()
        baseline_std = baseline_mean_and_std.select("Std").item()
        nrem_running_delta_avg = baseline_mean_and_std.select("Adaptive_Ld0_featureInputs_running_average").item()
        zscored_nrem_running_delta_avg = zscore_reward(nrem_running_delta_avg, baseline_mean, baseline_std)

        # Calculate time in NREM, and z-score it
        time_in_nrem = nrem_delta_power.height * SECONDS_PER_SAMPLE
        # Load baseline recording lengths statistics
        with open(baseline_rec_lengths_stats_path.format(
            participant=participant,
            device=device
        ), 'r') as f:
            baseline_rec_lengths_stats = json.load(f)
        
        # Extract mean and std for NREM time
        baseline_nrem_time_mean = baseline_rec_lengths_stats["mean"]
        baseline_nrem_time_std = baseline_rec_lengths_stats["std"]
        
        # Calculate z-scored time in NREM using the baseline statistics
        zscored_time_in_nrem = zscore_reward(time_in_nrem, baseline_nrem_time_mean, baseline_nrem_time_std)

        
        hemisphere_stats[side] = {
            "time_in_nrem": time_in_nrem,
            "zscored_time_in_nrem": zscored_time_in_nrem,
            "zscored_delta": zscored_nrem_running_delta_avg
        }
    
    
    # Calculate time-weighted reward
    if "Left" in hemisphere_stats and "Right" in hemisphere_stats:
        # Calculate total time in NREM across hemispheres
        total_nrem_time = (hemisphere_stats["Left"]["time_in_nrem"] + 
                      hemisphere_stats["Right"]["time_in_nrem"])
        
        # Calculate time-weighted delta reward
        if total_nrem_time > 0:  # Avoid division by zero
            delta_reward = (
                (hemisphere_stats["Left"]["time_in_nrem"] / total_nrem_time * 
                 hemisphere_stats["Left"]["zscored_delta"]) +
                (hemisphere_stats["Right"]["time_in_nrem"] / total_nrem_time * 
                 hemisphere_stats["Right"]["zscored_delta"])
            )
        else:
            delta_reward = 0.0
            
        # Calculate non-weighted average of zscored_time_in_nrem
        avg_zscored_time = (
            hemisphere_stats["Left"]["zscored_time_in_nrem"] + 
            hemisphere_stats["Right"]["zscored_time_in_nrem"]
        ) / 2
        
        # Apply (1-sigmoid)**(1/2) transformation to the zscored time average
        # This will act as a surrogate for Standard Error around the observation of average delta power
        sem = (1 - 1/(1 + np.exp(-avg_zscored_time)))**(1/2)
        
        # Final reward is the delta reward
        reward = delta_reward
    elif "Left" in hemisphere_stats:
        reward = hemisphere_stats["Left"]["zscored_delta"]
        sem = (1 - 1/(1 + np.exp(-hemisphere_stats["Left"]["zscored_time_in_nrem"])))**(1/2)
    elif "Right" in hemisphere_stats:
        reward = hemisphere_stats["Right"]["zscored_delta"]
        sem = (1 - 1/(1 + np.exp(-hemisphere_stats["Right"]["zscored_time_in_nrem"])))**(1/2)
    else:
        reward = 0.0
        sem = 0.0
        
    return {"bilaterally_weighted_average_NREM_delta_power": (reward, sem)}


def zscore_reward(
    reward: float,
    baseline_mean: float,
    baseline_std: float
) -> float:
    """
    Convert reward to z-score using baseline statistics.
    
    Z = (X - μ) / σ
    
    where:
    - X = reward value
    - μ = baseline mean
    - σ = baseline standard deviation
    
    Parameters
    ----------
    reward : float
        Raw reward value to be z-scored
    baseline_mean : float
        Mean of baseline distribution
    baseline_std : float
        Standard deviation of baseline distribution
        
    Returns
    -------
    float
        Z-scored reward value
    """
    return (reward - baseline_mean) / baseline_std


def observation_variance_calculation(
) -> float:
    """
    Calculate the variance of the observation. This is based up upon the length of the recording.
    """
    pass
