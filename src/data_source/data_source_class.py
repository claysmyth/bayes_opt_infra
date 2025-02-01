from src.utils import load_funcs
import polars as pl
import warnings


class DataSource:
    """
    A class to get data from a data source.
    Currently, only expects a single function to get data for each session to be run through the bayesian optimizer.
    Essentially, this assumes all sessions in a given pipeline are homogeneous,
    and data across all sessions can be collected with a singular function.

    The function should exist in the data_source_funcs module.

    If needed, a 'preprocessing' step can be added to this data source class,
    which would just include a chain of tasks, as defined in the data_source_config.
    """

    def __init__(self, data_source_config):
        self.config = data_source_config
        assert (
            self.config["functions"] is not None
        ), "functions must be defined in data_source_config"
        assert (
            len(self.config["functions"]) == 1
        ), "Only one get_data function is supported"
        self.func_name = list(self.config["functions"].keys())[0]
        self._get_data_task = load_funcs(self.config["functions"], "data_source")
        self._get_data = self._get_data_task[self.func_name]

    def get_data(self, trial: pl.DataFrame) -> pl.DataFrame:
        result = self._get_data(trial)
        if not isinstance(result, pl.DataFrame):
            warnings.warn(f"Expected polars DataFrame but got {type(result)}")
        return result
