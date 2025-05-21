from prefect import task
from src.utils import load_funcs
from typing import Dict, Any, List
import warnings
import polars as pl


class ParameterShipment:
    """
    A class to handle the shipment of parameters to their destination.
    Loads and executes shipment functions defined in the config.
    Each function should exist in the shipment_funcs module.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Load shipment functions
        assert config.get("shipment_functions") is not None, "shipment_functions must be defined in config"
        self.shipment_funcs = load_funcs(config["shipment_functions"], "shipment")

    def ship_parameters(self, parameters: Dict[str, Any] = None, participant_info: pl.DataFrame = None) -> Dict[str, List[Any]]:
        """
        Ship parameters using all configured shipment functions.
        
        Parameters
        ----------
        parameters : Dict[str, Any]
            Dictionary containing the parameters to ship
        participant_info : pl.DataFrame
            DataFrame containing participant information
            
        Returns
        -------
        Dict[str, List[Any]]
            Dictionary with function names as keys and their results as values
            Example: {
                "ship_rcs_adaptive_config": result1,
                "ship_other_config": result2
            }
        """
        results = {}
        
        for func_name, ship_func in self.shipment_funcs.items():
            # try:
            print(f"Running shipment function: {func_name}")
            result = ship_func(parameters, participant_info)
            results[func_name] = result
            # except Exception as e:
            #     warnings.warn(f"Error in shipment function '{func_name}': {str(e)}")
            #     raise
        
        return results
