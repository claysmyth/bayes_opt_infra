import polars as pl
import os
from omegaconf import DictConfig
from typing import List

# TODO: What shape should RCS incoming sessions be in?


class SessionManager:
    """Manages session data and tracks which sessions need to be processed.

    Attributes:
        sessions_config (DictConfig): Configuration for session management
        session_id_column (str): Name of column containing session IDs
        incoming_sessions_df (pl.DataFrame): DataFrame of all incoming sessions
        sessions_to_skip_dfs (List[pl.DataFrame]): List of DataFrames containing sessions to skip
        reported_sessions_df (pl.DataFrame): DataFrame of already reported sessions
        _tracked_sessions_df (pl.DataFrame): Internal copy of sessions with QA/QC status
    """

    def __init__(self, sessions_config: DictConfig) -> None:
        """Initialize SessionManager with config settings.

        Args:
            sessions_config: Configuration dictionary containing paths and settings
        """
        self.sessions_config = sessions_config

        # Check that all required keys are present.. otherwise, raise an error
        required_keys = [
            "SESSION_ID_COLUMN",
            "INCOMING_SESSIONS_CSV_PATH",
            "SESSIONS_TO_SKIP_CSV_PATHS",
            "REPORTED_SESSIONS_CSV_PATH",
            "FILTER_ON",
            "TO_INCLUDE",
        ]
        missing_keys = [key for key in required_keys if key not in sessions_config]
        assert (
            not missing_keys
        ), f"Required keys missing from sessions_config: {missing_keys}"

        # Get the primary key column for sessions to process
        self.session_id_column = sessions_config["SESSION_ID_COLUMN"]

        # Get the filter on column and the sessions to include
        self.filter_on_column = sessions_config["FILTER_ON"]
        self.sessions_to_include = sessions_config["TO_INCLUDE"]

        # Load and filter incoming sessions
        self.incoming_sessions_df = self._load_incoming_sessions()

        # Load sessions to skip
        self.sessions_to_skip_dfs = self._load_sessions_to_skip()

        # Filter out sessions that should be skipped
        self.incoming_sessions_df = self._get_new_sessions(
            self.incoming_sessions_df, self.sessions_to_skip_dfs
        )

        # Load previously reported sessions
        self.reported_sessions_df = self._load_reported_sessions()

        # Initialize tracked sessions DataFrame with QA/QC columns
        self._tracked_sessions_df = None

        # Get the csv preprocessing tasks to run on the incoming sessions csv (i.e. reformatting, filtering of specific session types or duplicates, etc...)
        self.tasks = self.sessions_config.get("tasks", None)

    def _load_incoming_sessions(self) -> pl.DataFrame:
        """Load and filter incoming sessions DataFrame."""
        df = pl.read_csv(self.sessions_config["INCOMING_SESSIONS_CSV_PATH"])

        df = df.filter(pl.col(self.filter_on_column).is_in(self.sessions_to_include))

        return df

    def _load_sessions_to_skip(self) -> List[pl.DataFrame]:
        """Load sessions that should be skipped from multiple CSV files."""
        skip_dfs = []
        for skip_path in self.sessions_config["SESSIONS_TO_SKIP_CSV_PATHS"]:
            if os.path.exists(skip_path):
                skip_dfs.append(pl.read_csv(skip_path))
        if len(skip_dfs) == 0:
            return pl.DataFrame()
        return skip_dfs

    def _load_reported_sessions(self) -> pl.DataFrame:
        """Load previously reported sessions."""
        reported_path = self.sessions_config["REPORTED_SESSIONS_CSV_PATH"]
        return (
            pl.read_csv(reported_path)
            if os.path.exists(reported_path)
            else pl.DataFrame()
        )

    def _get_new_sessions(
        self, df_to_filter: pl.DataFrame, dfs_to_filter_out: List[pl.DataFrame]
    ) -> pl.DataFrame:
        """Filter out sessions that should be excluded.

        Args:
            df_to_filter: DataFrame to filter
            dfs_to_filter_out: List of DataFrames containing sessions to exclude

        Returns:
            Filtered DataFrame with excluded sessions removed
        """
        filtered_df = df_to_filter
        for df_to_filter_out in dfs_to_filter_out:
            if not df_to_filter_out.is_empty():
                filtered_df = filtered_df.join(
                    df_to_filter_out, on=self.session_id_column, how="anti"
                )
        return filtered_df

    def _run_tasks(self, sessions: pl.DataFrame) -> pl.DataFrame:
        """Run tasks for a given set of sessions."""
        if self.tasks is None:
            return sessions
        for task in self.tasks:
            sessions = task(sessions)
        return sessions

    def get_new_sessions(self, run_tasks: bool = False) -> pl.DataFrame:
        """Get sessions that haven't been reported yet.

        Returns:
            DataFrame containing new sessions to process
        """
        new_sessions = self._get_new_sessions(
            self.incoming_sessions_df, [self.reported_sessions_df]
        )
        if run_tasks:
            new_sessions = self._run_tasks(new_sessions)
            
        # Create a copy with QA/QC status columns for tracking
        self._tracked_sessions_df = new_sessions.with_columns([
            pl.lit(False).alias("failed_qa"),
            pl.lit(False).alias("failed_qc"),
            pl.lit(None).cast(pl.Utf8).alias("failure_reason")
        ])
        
        return new_sessions

    def update_reported_sessions(self, sessions: pl.DataFrame, flags: List[str] = None) -> None:
        """Update the reported sessions DataFrame with new sessions."""

        if flags is not None and len(flags) > 0:
            sessions = sessions.with_columns([
                pl.lit(", ".join(flags)).alias("Flags")
            ])

        self.reported_sessions_df = pl.concat([self.reported_sessions_df, sessions], how='diagonal_relaxed')
        self._save_reported_sessions()

    def _save_reported_sessions(self) -> None:
        """Save reported sessions to a csv file."""
        self.reported_sessions_df.write_csv(
            self.sessions_config["REPORTED_SESSIONS_CSV_PATH"]
        )

    def mark_sessions_as_bad(self, sessions: pl.DataFrame) -> None:
        """Mark a session as bad (failed QA).
        
        Args:
            participant: Participant identifier matching session_id_column
        """
        sessions = sessions.with_columns([
            pl.lit(True).alias("failed_qa"),
            pl.lit("Failed quality assurance checks").alias("failure_reason")
        ])
        self.update_reported_sessions(sessions)
    
    
    def mark_sessions_as_insufficient(self, sessions: pl.DataFrame) -> None:
        """
        Mark a session as insufficient (failed QC).
        Args:
            participant: Participant identifier matching session_id_column
        """
        sessions = sessions.with_columns([
            pl.lit(True).alias("failed_qc"),
            pl.lit("Failed quality control checks").alias("failure_reason")
        ])
        self.update_reported_sessions(sessions)
