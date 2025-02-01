from prefect import task
from src.utils import load_funcs
from typing import Dict, Any
import warnings
from pathlib import Path
from string import Formatter


class ParameterShipment:
    """
    A class to handle the shipment of parameters to their destination.
    Currently, only expects a single function to ship parameters.
    The function should exist in the shipment_funcs module.
    """

    def __init__(self, shipment_config: Dict[str, Any]):
        self.config = shipment_config
        assert (
            self.config["functions"] is not None
        ), "functions must be defined in shipment_config"
        assert (
            len(self.config["functions"]) == 1
        ), "Only one shipment function is supported"
        self.func_name = list(self.config["functions"].keys())[0]
        self._ship_task = load_funcs(self.config["functions"], "shipment")
        self._ship = self._ship_task[self.func_name]

        # Set up paths
        self.base_dir = Path(self.config["paths"]["base_dir"])
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Get required template variables
        self.template_variables = self.config["functions"][self.func_name].get(
            "template_variables", []
        )

    @task(name="_format_path")
    def _format_path(self, path_template: str, variables: Dict[str, Any]) -> str:
        """Format path template with provided variables."""
        # Get all required variables from the template
        required_vars = {
            v[1] for v in Formatter().parse(path_template) if v[1] is not None
        }

        # Check if all required variables are provided
        missing_vars = required_vars - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        # Format the path
        return path_template.format(**variables)

    @task(name="ship_parameters_to_destination")
    def ship_parameters_to_destination(
        self, parameters: Dict[str, Any], template_vars: Dict[str, Any]
    ) -> str:
        """
        Ship parameters to their destination using template variables.

        Args:
            parameters: Dictionary of parameters from Bayesian optimization
            template_vars: Dictionary of variables to fill in template paths

        Returns:
            Path to the shipped file or confirmation message
        """
        # Validate template variables
        missing_vars = set(self.template_variables) - set(template_vars.keys())
        if missing_vars:
            raise ValueError(f"Missing required template variables: {missing_vars}")

        # Format source and destination paths
        source_path = self._format_path(
            self.config["shipment_template_source_path"], template_vars
        )
        dest_path = self._format_path(
            self.config["shipment_template_destination_path"], template_vars
        )

        # Ship parameters
        result = self._ship(parameters, source_path, dest_path, self.config)
        if not isinstance(result, str):
            warnings.warn(
                f"Expected string (file path or message) but got {type(result)}"
            )
        return result
