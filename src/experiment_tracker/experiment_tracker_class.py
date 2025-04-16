from ax.storage.sqa_store.db import get_engine, create_all_tables
from ax.storage.sqa_store.db import init_engine_and_session_factory
from ax.storage.sqa_store.structs import DBSettings
from ax.service.ax_client import AxClient
from src.utils import load_funcs
from pathlib import Path
import warnings
from typing import Dict, Any, Optional


class ExperimentTracker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_id_column = config.get("experiment_id_column", "experiment_id")
        
        # Base directory for all experiments
        self.base_dir = Path(config["paths"]["base_dir"])
        self.experiment_name = config["experiment"]["name"]
        
        # Store current context
        self.current_participant_id = None
        self.current_experiment_id = None
        self.ax_client = None
        self.current_trial_index = None  # Add this to track current trial
        self.current_parameters = None   # Add this to track current parameters

        # Load optimizer functions
        assert (
            config.get("optimizer_functions") is not None
        ), "optimizer_functions must be defined"
        self._optimizer_funcs = load_funcs(config["optimizer_functions"], "bayes_opt")
        self._complete_and_get_trial = self._optimizer_funcs["complete_and_get_trial"]
        
        # Load context update functions
        assert (
            config.get("context_update_functions") is not None
        ), "context_update_functions must be defined"
        self._context_funcs = load_funcs(config["context_update_functions"], "context_update")
        self._update_context_internal = self._context_funcs[config["context_update_functions"]["function"]]

        self.logger = get_run_logger()

    def _get_participant_json_path(self, participant_id: str) -> Path:
        """Get the JSON file path for a specific participant."""
        return self.base_dir / participant_id / f"{self.experiment_name}.json"

    def update_context(self, participant: Dict[str, Any], experiment_id: str) -> None:
        """
        Public method to update experimental context.
        Calls the configured context update function.
        """
        self._update_context_internal(self, participant, experiment_id)

    def _default_update_context(self, participant: Dict[str, Any], experiment_id: str) -> None:
        """Default implementation of context update."""
        participant_id = participant[self.experiment_id_column]
        
        # If we're already working with this participant, no need to reload
        if participant_id == self.current_participant_id:
            return
            
        # Update current context
        self.current_participant_id = participant_id
        self.current_experiment_id = experiment_id
        
        # Load participant-specific Ax client
        self.ax_client = self._load_ax_client(participant_id)

    def _load_ax_client(self, participant_id: str) -> AxClient:
        """Load Ax client from JSON or database for a specific participant."""
        if self.config["experiment"].get("use_database", False):
            db_path = self.base_dir / participant_id / f"{self.experiment_name}.db"
            if not db_path.exists():
                raise FileNotFoundError(
                    f"Database not found at {db_path}. "
                    "Please initialize experiment using initialize_experiment.py"
                )
            return AxClient(db_settings=DBSettings(url=f"sqlite:///{db_path}"))
        else:
            json_path = self._get_participant_json_path(participant_id)
            if not json_path.exists():
                raise FileNotFoundError(
                    f"JSON file not found at {json_path}. "
                    "Please initialize experiment using initialize_experiment.py"
                )
            return AxClient.load_from_json_file(filepath=str(json_path))

    def save_current_state(self) -> None:
        """Save the current state of the Ax client."""
        if self.ax_client is not None and self.current_participant_id is not None:
            json_path = self._get_participant_json_path(self.current_participant_id)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            self.ax_client.save_to_json_file(filepath=str(json_path))

    def get_next_trial(self) -> Dict[str, Any]:
        """
        Get parameters for the next trial.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing 'parameters' and 'trial_index'
        """
        if self.ax_client is None:
            raise RuntimeError("No active Ax client. Call update_context first.")
        
        parameters, trial_index = self.ax_client.get_next_trial()
        
        # Store current trial info
        self.current_trial_index = trial_index
        self.current_parameters = parameters
        
        # Save state after getting new trial
        self.save_current_state()
        
        return {
            "parameters": parameters,
            "trial_index": trial_index
        }

    def update_optimizer(self, evaluation_result: Dict[str, float]) -> Dict[str, Any]:
        """
        Complete current trial with results and get parameters for next trial.
        
        Parameters
        ----------
        evaluation_result : Dict[str, float]
            Dictionary containing the evaluation metrics
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing parameters for next trial
        """
        if self.ax_client is None:
            raise RuntimeError("No active Ax client. Call update_context first.")
        
        if self.current_trial_index is None:
            raise RuntimeError("No active trial. Call get_next_trial first.")
        
        # Complete the current trial
        self.ax_client.complete_trial(
            trial_index=self.current_trial_index,
            raw_data=evaluation_result
        )
        
        # Save state after completing trial
        self.save_current_state()
        
        # Get parameters for next trial (this also saves state)
        next_trial = self.get_next_trial()
        
        return next_trial