import polars as pl
from typing import Dict, Tuple


def calculate_bilateral_reward(
    session_data_dict: Dict[str, pl.DataFrame],
    delta_power_col: Dict[str, str] = {"Left": "Power_Band5", "Right": "Power_Band5"},
    nrem_state: Dict[str, str] = {"Left": "State 1", "Right": "State 1"}
) -> float:
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
    delta_power_col : Dict[str, str], optional
        Dictionary specifying delta power column name for each hemisphere
    nrem_state : Dict[str, str], optional
        Dictionary specifying NREM state label for each hemisphere
        
    Returns
    -------
    float
        Bilateral reward value
    """
    # Calculate time in NREM and average delta power for each hemisphere
    hemisphere_stats = {}
    for side in ["Left", "Right"]:
        data = session_data_dict[side]
        
        # Get data in NREM state
        nrem_data = data.filter(pl.col("Adaptive_CurrentAdaptiveState") == nrem_state[side])
        
        # Calculate time in NREM (number of samples)
        time_in_nrem = len(nrem_data)
        
        # Calculate average delta power in NREM
        avg_delta = nrem_data.select(pl.col(delta_power_col[side]).mean()).item()
        
        hemisphere_stats[side] = {
            "time_in_nrem": time_in_nrem,
            "avg_delta": avg_delta
        }
    
    # Calculate total time in NREM across hemispheres
    total_nrem_time = (hemisphere_stats["Left"]["time_in_nrem"] + 
                      hemisphere_stats["Right"]["time_in_nrem"])
    
    # Calculate time-weighted reward
    if total_nrem_time > 0:  # Avoid division by zero
        reward = (
            (hemisphere_stats["Left"]["time_in_nrem"] / total_nrem_time * 
             hemisphere_stats["Left"]["avg_delta"]) +
            (hemisphere_stats["Right"]["time_in_nrem"] / total_nrem_time * 
             hemisphere_stats["Right"]["avg_delta"])
        )
    else:
        reward = 0.0
    
    return reward


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
