import polars as pl
from typing import Dict


def sleep_delta_reward(
    session_data: pl.DataFrame,
    delta_power_col: str = "Power_Band5",
    nrem_state: str = "State 1",
):
    """
    Calculate average delta power when in N2/3 sleep.
    Assumes Power_Band5 corresponds to cortical delta power, and State 1 corresponds to NREM sleep.
    """
    # Forward fill state and amplitude columns to handle missing values
    session_data = session_data.with_columns(
        [
            pl.col("Adaptive_CurrentAdaptiveState").fill_null(strategy="forward"),
            pl.col("Adaptive_CurrentProgramAmplitudesInMilliamps_1").fill_null(
                strategy="forward"
            ),
        ]
    )

    # Get mode of stimulation amplitude during State 1. This infers target amplitude for NREM sleep.
    mode_amp = (
        session_data.filter(pl.col("Adaptive_CurrentAdaptiveState") == nrem_state)
        .select(pl.col("Adaptive_CurrentProgramAmplitudesInMilliamps_1").mode())
        .item()
    )

    # Filter for State 1 and mode amplitude, then get mean delta power. This collects mean delta power for target amplitude of NREM sleep.
    average_delta = (
        session_data.filter(
            (pl.col("Adaptive_CurrentAdaptiveState") == nrem_state)
            & (pl.col("Adaptive_CurrentProgramAmplitudesInMilliamps_1") == mode_amp)
        )
        .select(pl.col(delta_power_col).mean())
        .item()
    )

    return average_delta
