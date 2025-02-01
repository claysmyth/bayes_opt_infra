from src.utils import load_funcs
import polars as pl
import warnings
from typing import Dict, Any, Optional


class Evaluation:
    """
    A class to evaluate optimization objectives from session data.
    Currently, only expects a single evaluation function for each experiment.
    The evaluation function should exist in the evaluation_funcs module.
    """

    def __init__(self, evaluation_config: Dict[str, Any]):
        self.config = evaluation_config
        assert (
            self.config["functions"] is not None
        ), "functions must be defined in evaluation_config"
        assert (
            len(self.config["functions"]) == 1
        ), "Only one evaluation function is supported"
        self.func_name = list(self.config["functions"].keys())[0]
        self._evaluate_task = load_funcs(self.config["functions"], "evaluation")
        self._evaluate = self._evaluate_task[self.func_name]
        self.last_result: Optional[Dict[str, float]] = None

    def evaluate(self, session_data: pl.DataFrame) -> Dict[str, float]:
        """
        Evaluate objective(s) from session data.
        Returns a dictionary with the objective name(s) and value(s).
        """
        result = self._evaluate(session_data)
        if not isinstance(result, dict):
            warnings.warn(f"Expected dictionary but got {type(result)}")
        self.last_result = result
        return result
