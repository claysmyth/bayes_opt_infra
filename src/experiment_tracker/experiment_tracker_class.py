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

        # Load Ax client from file
        self.base_dir = Path(config["paths"]["base_dir"])
        self.experiment_name = config["experiment"]["name"]
        self.ax_client = self._load_ax_client()

        # Load optimizer functions
        assert (
            config.get("optimizer_functions") is not None
        ), "optimizer_functions must be defined"
        self._optimizer_funcs = load_funcs(config["optimizer_functions"], "bayes_opt")
        self._complete_and_get_trial = self._optimizer_funcs["complete_and_get_trial"]

        self.experiments = {}  # Dict to store experiment metadata
        self.logger = get_run_logger()

    def _load_ax_client(self) -> AxClient:
        """Load Ax client from JSON or database"""
        if self.config["experiment"].get("use_database", False):
            db_path = self.base_dir / f"{self.experiment_name}.db"
            if not db_path.exists():
                raise FileNotFoundError(f"Database not found at {db_path}")
            # Load from database
            return AxClient(db_settings=DBSettings(url=f"sqlite:///{db_path}"))
        else:
            json_path = self.base_dir / f"{self.experiment_name}.json"
            if not json_path.exists():
                raise FileNotFoundError(f"JSON file not found at {json_path}")
            # Load from JSON
            return AxClient.load_from_json_file(filepath=str(json_path))

    def update_optimizer(self, evaluation_result: Dict[str, float]) -> Dict[str, Any]:
        """Update optimizer with evaluation results and get next parameters"""
        return self._complete_and_get_trial(self, evaluation_result)

    # def get_or_create_experiment(self, experiment_id):
    #     """Get or create an experiment instance"""
    #     if experiment_id not in self.experiments:
    #         self.experiments[experiment_id] = {
    #             'participants': {},
    #             'metadata': {},
    #             'creation_date': datetime.now(),
    #             'status': 'active'
    #         }
    #     return self.experiments[experiment_id]

    # def add_session(self, experiment_id, participant_id, session_data):
    #     """Add a new session to the experiment/participant history"""
    #     experiment = self.get_or_create_experiment(experiment_id)
    #     if participant_id not in experiment['participants']:
    #         experiment['participants'][participant_id] = {
    #             'sessions': [],
    #             'current_parameters': None,
    #             'optimization_history': [],
    #             'status': 'active'
    #         }

    #     experiment['participants'][participant_id]['sessions'].append(session_data)

    # def get_optimization_state(self, experiment_id, participant_id):
    #     """Get current optimization state for a participant"""
    #     return self.experiments[experiment_id]['participants'][participant_id]['optimization_history']
