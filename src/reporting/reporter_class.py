import wandb
import polars as pl
import os
import warnings
from src.experiment_tracker.experiment_tracker_class import ExperimentTracker
from .reporter_utils import local_setup, log_plotting_result
from prefect import task
from typing import Dict, Any, Optional
from src.utils import load_funcs

# NOTE: Reporter is different than the reported_sessions_csv.
# Reporter is the class that handles the reporting of sessions to wandb and local directories.
# The reported_sessions_csv is a csv file that contains the sessions that have been processed for optimization.
class Reporter:
    def __init__(self, reporting_config: Dict[str, Any]):
        """
        Initialize the Reporter with configuration.
        
        Parameters
        ----------
        reporting_config : Dict[str, Any]
            Configuration dictionary containing:
            - local_reporting_func: Configuration for local reporting
            - wandb_init_func: Configuration for Weights & Biases initialization
            - report_results_functions: List of functions to run for reporting results
            - log_shipment_functions: List of functions to run for logging shipments
        """
        self.config = reporting_config
        self._init_local_reporting = None
        self._init_wandb = None
        self._viz_funcs = None
        self._reporting_objects = None
        
        # Initialize functions from config
        self._initialize_functions()
        
    def _initialize_functions(self):
        """Initialize functions based on config."""
        # Initialize local reporting function
        if "local_reporting_func" in self.config:
            local_func_name = list(self.config["local_reporting_func"].keys())[0]
            self._init_local_reporting = load_funcs(self.config["local_reporting_func"], "reporter", return_type="handle")
            
        # Initialize wandb function
        if "wandb_init_func" in self.config:
            wandb_func_name = list(self.config["wandb_init_func"].keys())[0]
            self._init_wandb = load_funcs(self.config["wandb_init_func"], "reporter", return_type="handle")
        
        # Initialize log shipment function
        if "log_shipment_func" in self.config:
            log_shipment_func_name = list(self.config["log_shipment_func"].keys())[0]
            self._log_shipment = load_funcs(self.config["log_shipment_func"], "reporter", return_type="handle")
            
        # Load visualization functions
        self._viz_func_config = {func_name: func_config.get("kwargs", {}) for func_name, func_config in self.config["report_results_functions"].items()}

        
    def _format_and_run_local_init(self, session_info: pl.DataFrame, experiment_tracker: Any) -> Optional[str]:
        """
        Format and run local initialization for a specific participant.
        
        Parameters
        ----------
        session_info : pl.DataFrame
            DataFrame containing session information
        experiment_tracker : Any
            Experiment tracker object to get current trial index
            
        Returns
        -------
        Optional[str]
            Path to local reporting directory if initialized, None otherwise
        """
        if not self._init_local_reporting:
            return None
            
        # Get reporting path base from config
        reporting_path_base = self.config["local_reporting_func"][list(self.config["local_reporting_func"].keys())[0]]["reporting_path_base"]
        
        # Initialize local reporting with trial path
        return self._init_local_reporting(
            session_info,
            experiment_tracker,
        )
        
    def _format_and_run_wandb_init(self, session_info: pl.DataFrame, local_path: Optional[str] = None):
        """
        Format and run wandb initialization for a specific participant.
        
        Parameters
        ----------
        session_info : pl.DataFrame
            DataFrame containing session information
        local_path : Optional[str]
            Path to local reporting directory
            
        Returns
        -------
        Optional[wandb.Run]
            wandb run object if initialized, None otherwise
        """
        if not self._init_wandb:
            return None
            
        # Get wandb config
        wandb_config = self.config["wandb_init_func"][list(self.config["wandb_init_func"].keys())[0]]
        
        # Add session and device info to config
        wandb_config["tags"] = session_info.get_column("Session#").unique().to_list()
        wandb_config["job_type"] = session_info.get_column("Device").unique().to_list()[0]
        wandb_config["group"] = session_info.get_column("SessionType(s)").unique().to_list()[0]
        
        # Initialize wandb
        run = wandb.init(
            project=wandb_config["project"],
            entity=wandb_config["entity"],
            job_type=wandb_config["job_type"],
            group=wandb_config["group"],
            tags=wandb_config["tags"],
        )
        
        return run
        
    def update_for_current_participant(self, session_info: pl.DataFrame, experiment_tracker: Any) -> None:
        """
        Initialize reporting for current participant and store reporting objects.
        
        Parameters
        ----------
        session_info : pl.DataFrame
            DataFrame containing session information
        experiment_tracker : Any
            Experiment tracker object to get current trial index
        """
        # Store session
        self._session_info = session_info

        # Initialize local reporting
        local_path = self._format_and_run_local_init(session_info, experiment_tracker)
        
        # Initialize wandb
        wandb_run = self._format_and_run_wandb_init(session_info, local_path)
        
        # Store reporting objects
        self._reporting_objects = {
            "local_path": local_path,
            "wandb_run": wandb_run
        }

        # Initialize visualization functions. Add session info to kwargs.
        participant_viz_func_config = {func_name: func_config | {"session_info": self._session_info} for func_name, func_config in self._viz_func_config.items()}
        self._viz_funcs = load_funcs(participant_viz_func_config, "visualization", return_type="dict")

    
    def reinit_local_reporting(self, session_info: pl.DataFrame, experiment_tracker: Any) -> None:
        """
        Reinitialize local reporting for a specific participant.
        
        Parameters
        ----------
        session_info : pl.DataFrame
        """
        local_path = self._format_and_run_local_init(session_info, experiment_tracker)
        self._reporting_objects["local_path"] = local_path

        
    def report_results(
        self,
        session_data: Dict[str, pl.DataFrame],
        result: Dict[str, Any],
        participant_sessions: pl.DataFrame,
        experiment_tracker: ExperimentTracker
    ) -> Optional[str]:
        """
        Report results using configured functions.
        
        Parameters
        ----------
        session_data : Dict[str, pl.DataFrame]
            Dictionary containing DataFrames with different types of session data
        result : Dict[str, Any]
            Dictionary containing evaluation results
        participant_sessions : pl.DataFrame
            DataFrame containing participant session information
        experiment_tracker : ExperimentTracker
            Experiment tracker object
            
        Returns
        -------
        Optional[str]
            Path to local reporting directory if used, None otherwise
        """
        if self._reporting_objects is None:
            raise ValueError("Reporting objects not initialized. Call update_for_current_participant first.")
            
        local_path = self._reporting_objects.get("local_path")
        wandb_run = self._reporting_objects.get("wandb_run")
        
        # Execute configured functions
        if "report_results_functions" in self.config:
            for func_name, func_config in self.config["report_results_functions"].items():
                # Get function parameters
                data_source = func_config.get("data", "dataframe")
                log_options = func_config.get("log", [])
                func_params = func_config.get("kwargs", {})
                
                # Get data based on source
                if data_source == "dataframe":
                    # Apply function to each DataFrame in the dictionary
                    for data_name, data_df in session_data.items():
                        # Execute visualization function
                        viz_func = self._viz_funcs.get(func_name, None)
                        if viz_func:
                            #try:
                                # Add data name to parameters for identification
                                # result = viz_func(data_df, **{**func_params, "data_name": data_name})
                            result = viz_func(data_df)
                            log_plotting_result(result, f"{func_name}_{data_name}", log_options, wandb_run, local_path)
                            # except Exception as e:
                            #     print(f"Error executing {func_name} on {data_name}: {e}")
                        else:
                            print(f"Warning: Function '{func_name}' not found in viz_funcs")
                elif data_source == "result":
                    data = result
                elif data_source == "sessions":
                    data = participant_sessions
                elif data_source == "experiment_tracker" or data_source == "experiment":
                    data = experiment_tracker
                else:
                    print(f"Warning: Unknown data source '{data_source}' for function '{func_name}'")
                    continue
                
                # Execute visualization function for non-dataframe sources
                if data_source != "dataframe":
                    viz_func = self._viz_funcs.get(func_name, None)
                    if viz_func:
                        try:
                            result = viz_func(data, **func_params)
                            log_plotting_result(result, func_name, log_options, wandb_run, local_path)
                        except Exception as e:
                            print(f"Error executing {func_name}: {e}")
                    else:
                        print(f"Warning: Function '{func_name}' not found in viz_funcs")
        
        # Finish wandb run if initialized
        if wandb_run:
            wandb_run.finish()
            
        return local_path
        
    def log_shipment(
        self,
        shipped_files: Dict[str, Any],
        experiment_tracker: ExperimentTracker
    ) -> None:
        """
        Log shipment information using configured functions.
        
        Parameters
        ----------
        next_trial : Dict[str, Any]
            Dictionary containing next trial parameters
        shipped_files : Dict[str, Any]
            Dictionary containing shipped files information
        """
        
        if self._reporting_objects is None:
            raise ValueError("Reporting objects not initialized. Call update_for_current_participant first.")
            
        local_path = self._reporting_objects.get("local_path")
        self._log_shipment(shipped_files, experiment_tracker, local_path)
