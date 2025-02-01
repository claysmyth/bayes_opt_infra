import polars as pl
import os
from omegaconf import DictConfig

# TODO: What shape should RCS incoming sessions be in?


class SessionManager:
    """Manages session data and tracks which sessions need to be processed.

    Attributes:
        sessions_config (DictConfig): Configuration for session management
        session_id_column (str): Name of column containing session IDs
        incoming_sessions_df (pl.DataFrame): DataFrame of all incoming sessions
        sessions_to_skip_df (pl.DataFrame): DataFrame of sessions to skip
        reported_sessions_df (pl.DataFrame): DataFrame of already reported sessions
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
            "SESSIONS_TO_SKIP_CSV_PATH",
            "REPORTED_SESSIONS_CSV_PATH",
            "SESSION_TYPES",
        ]
        missing_keys = [key for key in required_keys if key not in sessions_config]
        assert (
            not missing_keys
        ), f"Required keys missing from sessions_config: {missing_keys}"

        # Get the primary key column for sessions to process
        self.session_id_column = sessions_config["SESSION_ID_COLUMN"]

        # Load and filter incoming sessions
        self.incoming_sessions_df = self._load_incoming_sessions()

        # Load sessions to skip
        self.sessions_to_skip_df = self._load_sessions_to_skip()

        # Filter out sessions that should be skipped
        self.incoming_sessions_df = self._get_new_sessions(
            self.incoming_sessions_df, self.sessions_to_skip_df
        )

        # Load previously reported sessions
        self.reported_sessions_df = self._load_reported_sessions()

        # Get the csv preprocessing tasks to run on the incoming sessions csv (i.e. reformatting, filtering of specific session types or duplicates, etc...)
        self.tasks = self.sessions_config.get("tasks", None)

    def _load_incoming_sessions(self) -> pl.DataFrame:
        """Load and filter incoming sessions DataFrame."""
        df = pl.read_csv(self.sessions_config["INCOMING_SESSIONS_CSV_PATH"])

        if session_types := self.sessions_config.get("SESSION_TYPES"):
            df = df.filter(pl.col("Session_Type").is_in(session_types))

        return df

    def _load_sessions_to_skip(self) -> pl.DataFrame:
        """Load sessions that should be skipped."""
        skip_path = self.sessions_config["SESSIONS_TO_SKIP_CSV_PATH"]
        return pl.read_csv(skip_path) if os.path.exists(skip_path) else pl.DataFrame()

    def _load_reported_sessions(self) -> pl.DataFrame:
        """Load previously reported sessions."""
        reported_path = self.sessions_config["REPORTED_SESSIONS_CSV_PATH"]
        return (
            pl.read_csv(reported_path)
            if os.path.exists(reported_path)
            else pl.DataFrame()
        )

    def _get_new_sessions(
        self, df_to_filter: pl.DataFrame, df_to_filter_out: pl.DataFrame
    ) -> pl.DataFrame:
        """Filter out sessions that should be excluded.

        Args:
            df_to_filter: DataFrame to filter
            df_to_filter_out: DataFrame containing sessions to exclude

        Returns:
            Filtered DataFrame with excluded sessions removed
        """
        if df_to_filter_out.is_empty():
            return df_to_filter
        return df_to_filter.join(
            df_to_filter_out, on=self.session_id_column, how="anti"
        )

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
            self.incoming_sessions_df, self.reported_sessions_df
        )
        if run_tasks:
            new_sessions = self._run_tasks(new_sessions)
        return new_sessions

    def update_reported_sessions(self, sessions: pl.DataFrame) -> None:
        """Update the reported sessions DataFrame with new sessions."""
        self.reported_sessions_df = pl.concat([self.reported_sessions_df, sessions])
        self._save_reported_sessions()

    def _save_reported_sessions(self) -> None:
        """Save reported sessions to a csv file."""
        self.reported_sessions_df.write_csv(
            self.sessions_config["REPORTED_SESSIONS_CSV_PATH"]
        )
