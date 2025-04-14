import polars as pl
from typing import Dict


def sleep_delta_reward_unilateral(
    session_data: pl.DataFrame,
    delta_power_col: str = "Power_Band5",
    nrem_state: str = "State 1",
):
    """Calculate average delta power when in N2/3 sleep."""
    # Forward fill state and amplitude columns
    session_data = session_data.with_columns(
        [
            pl.col("Adaptive_CurrentAdaptiveState").fill_null(strategy="forward"),
            pl.col("Adaptive_CurrentProgramAmplitudesInMilliamps_1").fill_null(
                strategy="forward"
            ),
        ]
    )

    # Get mode amplitude during NREM
    mode_amp = (
        session_data.filter(pl.col("Adaptive_CurrentAdaptiveState") == nrem_state)
        .select(pl.col("Adaptive_CurrentProgramAmplitudesInMilliamps_1").mode())
        .item()
    )

    # Get mean delta power during NREM at mode amplitude
    average_delta = (
        session_data.filter(
            (pl.col("Adaptive_CurrentAdaptiveState") == nrem_state)
            & (pl.col("Adaptive_CurrentProgramAmplitudesInMilliamps_1") == mode_amp)
        )
        .select(pl.col(delta_power_col).mean())
        .item()
    )

    return {"average_NREM_delta_power": average_delta}


def sleep_delta_reward_unilateral_standardized(
    session_data: pl.DataFrame,
    delta_power_col: str = "Power_Band5",
    nrem_state: str = "State 1",
    standardization_params: Dict[str, float] = None,
):
    """
    Calculate standardized average delta power when in N2/3 sleep.
    
    Args:
        session_data: DataFrame containing session data
        delta_power_col: Column name for delta power
        nrem_state: State name for NREM sleep
        standardization_params: Dict containing 'mean' and 'std' for standardization
            If None, returns unstandardized value
    """
    # Get raw average delta
    result = sleep_delta_reward_unilateral(session_data, delta_power_col, nrem_state)
    raw_value = result["average_NREM_delta_power"]
    
    # Standardize if parameters provided
    if standardization_params is not None:
        mean = standardization_params.get("mean", 0)
        std = standardization_params.get("std", 1)
        standardized_value = (raw_value - mean) / std
        return_dict = {"average_NREM_delta_power": standardized_value}
    else:
        return_dict = {"average_NREM_delta_power": raw_value}
    
    return return_dict


def sleep_delta_reward_bilateral(
    session_data_dict: Dict[str, pl.DataFrame],
    delta_power_col: Dict[str, str] = {"Left": "Power_Band5", "Right": "Power_Band5"},
    nrem_state: Dict[str, str] = {"Left": "State 1", "Right": "State 1"},
    null_check_col: Dict[str, str] = {"Left": "TD_key3", "Right": "TD_key3"},
):
    """Calculate bilateral average delta power when in N2/3 sleep."""
    reward_dict = {
        key: list(sleep_delta_reward_unilateral(
            session_data_dict[key], delta_power_col[key], nrem_state[key]
        ).values())[0]
        for key in session_data_dict.keys()
    }

    # Calculate weights based on non-null data length
    non_null_len = {
        key: session_data_dict[key]
        .filter(pl.col(null_check_col[key]).is_not_null())
        .shape[0]
        for key in session_data_dict.keys()
    }
    non_null_len_sum = sum(non_null_len.values())
    reward_weight = {
        key: non_null_len[key] / non_null_len_sum
        for key in session_data_dict.keys()
    }

    # Calculate weighted rewards
    reward_dict = {
        key: reward_dict[key] * reward_weight[key] for key in session_data_dict.keys()
    }
    reward = sum(reward_dict.values())
    
    return {"average_NREM_delta_power": reward}


def sleep_delta_reward_bilateral_standardized(
    session_data_dict: Dict[str, pl.DataFrame],
    delta_power_col: Dict[str, str] = {"Left": "Power_Band5", "Right": "Power_Band5"},
    nrem_state: Dict[str, str] = {"Left": "State 1", "Right": "State 1"},
    null_check_col: Dict[str, str] = {"Left": "TD_key3", "Right": "TD_key3"},
    standardization_params: Dict[str, Dict[str, float]] = None,
):
    """
    Calculate standardized bilateral average delta power when in N2/3 sleep.
    
    Args:
        session_data_dict: Dictionary of session data for each hemisphere
        delta_power_col: Dictionary of column names for delta power
        nrem_state: Dictionary of state names for NREM sleep
        null_check_col: Dictionary of column names for null check
        standardization_params: Dictionary of standardization parameters for each hemisphere
            Example: {
                "Left": {"mean": 0.5, "std": 0.1},
                "Right": {"mean": 0.6, "std": 0.12}
            }
            If None, returns unstandardized values
    """
    # Get standardized rewards for each hemisphere
    reward_dict = {
        key: list(sleep_delta_reward_unilateral_standardized(
            session_data_dict[key],
            delta_power_col[key],
            nrem_state[key],
            standardization_params.get(key) if standardization_params else None
        ).values())[0]
        for key in session_data_dict.keys()
    }

    # Calculate weights based on non-null data length
    non_null_len = {
        key: session_data_dict[key]
        .filter(pl.col(null_check_col[key]).is_not_null())
        .shape[0]
        for key in session_data_dict.keys()
    }
    non_null_len_sum = sum(non_null_len.values())
    reward_weight = {
        key: non_null_len[key] / non_null_len_sum
        for key in session_data_dict.keys()
    }

    # Calculate weighted rewards
    reward_dict = {
        key: reward_dict[key] * reward_weight[key] for key in session_data_dict.keys()
    }
    reward = sum(reward_dict.values())
    
    return {"average_NREM_delta_power": reward}
