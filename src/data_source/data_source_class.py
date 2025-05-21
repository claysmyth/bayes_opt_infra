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
            self.config["get_data_function"] is not None
        ), "get_data_function must be defined in data_source_config"
        assert (
            len(self.config["get_data_function"]) == 1
        ), "Only one get_data function is supported"
        self.func_name = list(self.config["get_data_function"].keys())[0]
        self._get_data = load_funcs(self.config["get_data_function"], "data_source", return_type="handle")

    def get_data(self, trial: pl.DataFrame) -> pl.DataFrame:
        result = self._get_data(trial)
        if not isinstance(result, pl.DataFrame | dict):
            warnings.warn(f"Expected polars DataFrame or dict but got {type(result)}")
        return result
