import wandb
import polars as pl
import pandas as pd
import os
import warnings

# NOTE: Reporter is different than the reported_sessions_csv. 
# Reporter is the class that handles the reporting of sessions to wandb and local directories.
# The reported_sessions_csv is a csv file that contains the sessions that have been processed for optimization.
class Reporter:
    def __init__(self, reporting_config):
        self.config = reporting_config


    def init_wandb(self, session_info: pl.DataFrame, path: str):
        # Add session and device relevant info to wandb_config for filtering in dashboard
        self.wandb_config["tags"] = (
            session_info.get_column("Session#").unique().to_list()
        )
        self.wandb_config["job_type"] = (
            session_info.get_column("Device").unique().to_list()[0]
        )
        self.wandb_config["group"] = (
            session_info.get_column("SessionType(s)").unique().to_list()[0]
        )

        # Initialize wandb
        run = wandb.init(
            project=self.wandb_config["project"],
            entity=self.wandb_config["entity"],
            job_type=self.wandb_config["job_type"],
            group=self.wandb_config["group"],
            tags=self.wandb_config["tags"],
        )

        # Save all session info to wandb
        run.log(
            {
                "session_info": wandb.Table(
                    dataframe=session_info.select(
                        pl.exclude("Data_Server_Hyperlink")
                    # Polars to pandas conversion can mess up timezone info, so explicitly set it
                    ).to_pandas().assign(TimeStarted=lambda df: pd.to_datetime(df['TimeStarted']).dt.tz_localize(self.config["TIMEZONE"]))
                )
            }
        )
        if path:
            run.log({"path_to_local_reports": path})

        # Log settings CSV files for each session in grouping
        tables = {}
        for i in range(session_info.height):
            csv_dir = session_info[i, "csv_path"]
            if os.path.isdir(csv_dir):
                for filename in os.listdir(csv_dir):
                    if filename.endswith(".csv"):
                        csv_path = os.path.join(csv_dir, filename)
                        try:
                            df = pd.read_csv(csv_path)
                            table_name = os.path.splitext(filename)[
                                0
                            ]  # Remove .csv extension
                            if table_name not in tables:
                                tables[table_name] = []
                            tables[table_name].append(df)
                        except Exception as e:
                            warnings.warn(
                                f"Error reading CSV file {csv_path}: {str(e)}"
                            )
            else:
                warnings.warn(f"CSV directory not found: {csv_dir}")

        # Concatenate DataFrames and log as wandb Tables
        for table_name, dfs in tables.items():
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                run.log({table_name: wandb.Table(dataframe=combined_df)})

        return run

    def init_local_reporting(self, session_info):
        # Create local directory structure
        rcs_num = session_info.get_column("Device").unique().to_list()[0]
        session_type = session_info.get_column("SessionType(s)").unique().to_list()[0]
        session_num = "_".join(session_info.get_column("Session#").unique().to_list())
        reporting_path = os.path.join(
            self.local_reporting_config["reporting_path_base"],
            session_type,
            rcs_num,
            "session_reports",
            session_num,
        )
        if not os.path.exists(reporting_path):
            os.makedirs(reporting_path)
        else:
            warnings.warn(f"Directory already exists: {reporting_path}", UserWarning)

        # Code snapshot, git info, and conda package versions saved to local directory
        local_setup(reporting_path, self.local_reporting_config, conda=False)
        return reporting_path