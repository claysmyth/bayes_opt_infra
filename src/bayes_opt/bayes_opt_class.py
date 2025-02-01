import polars as pl
import yaml
from typing import Dict, Any, Callable
import importlib

# from configs_and_globals.configs import visualization_config
import src.bayes_opt.bayes_funcs as bayes
from src.viz_and_reports.reporting_funcs import log_plotting_result
from prefect import task, flow
import wandb
import os
import warnings
from src.viz_and_reports.reporting_funcs import local_setup
import pandas as pd


def load_function(function_path: str) -> Callable:
    """
    Dynamically load a function from a string path.
    """
    module_name, function_name = function_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


class BayesOpt:
    def __init__(self, bayes_opt_config):
        self.config = bayes_opt_config
        self.bayes_funcs = {
            func_name: getattr(bayes, func_name, None)
            for func_name in self.config["functions"]
        }
        self.wandb_config = self.config["wandb"]
        self.local_reporting_config = self.config["local_reporting"]
