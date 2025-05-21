from src.utils import load_funcs
import polars as pl
import warnings
import json
from typing import Dict, Any, Optional
from pathlib import Path


class Evaluation:
    """
    A class to evaluate optimization objectives from session data.
    Currently, only expects a single evaluation function for each experiment.
    The evaluation function should exist in the evaluation_funcs module.
    """

    def __init__(self, evaluation_config: Dict[str, Any]):
        self.config = evaluation_config
        assert (
            self.config["evaluation_function"] is not None
        ), "functions must be defined in evaluation_config"
        
        # Load supporting parameters paths
        self.supporting_params_paths = self.config.get("supporting_params", {})
        self.supporting_params = {}
        
        # Initialize evaluation function
        self.func_name = list(self.config["evaluation_function"].keys())[0]
        self._evaluate_task = load_funcs(self.config["evaluation_function"], "evaluation")
        self._evaluate = self._evaluate_task[self.func_name]
        self.last_result: Optional[Dict[str, float]] = None

    # def _load_supporting_params(self, participant: str, device: str) -> None:
    #     """Load supporting parameters for the current participant and device."""
    #     for param_name, path_template in self.supporting_params_paths.items():
    #         path = path_template.format(participant=participant, device=device)
    #         try:
    #             with open(path, 'r') as f:
    #                 self.supporting_params[param_name] = json.load(f)
    #         except Exception as e:
    #             warnings.warn(f"Failed to load supporting parameter {param_name} from {path}: {str(e)}")

    def evaluate(self, session_data: pl.DataFrame, participant: str) -> Dict[str, float]:
        """
        Evaluate objective(s) from session data.
        Returns a dictionary with the objective name(s) and value(s).
        """
        # Get participant and device info from session data
        # participant = session_data["participant"].unique()[0]
        # device = session_data["device"].unique()[0]
        
        # Load supporting parameters if paths are configured
        # if self.supporting_params_paths:
        #     self._load_supporting_params(participant, device)
        
        # Pass both session data and supporting params to evaluation function
        result = self._evaluate(session_data, participant)
        if not isinstance(result, dict):
            warnings.warn(f"Expected dictionary but got {type(result)}")
        self.last_result = result
        return result
