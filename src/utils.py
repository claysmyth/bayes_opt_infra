from prefect import task
import src.analysis.analysis_funcs as analysis_funcs
import src.data_source.data_source_funcs as data_source_funcs
import src.quality_assurance_control.quality_assurance as quality_assurance_funcs
import src.quality_assurance_control.quality_control as quality_control_funcs
import src.evaluation.evaluation_funcs as evaluate_funcs
import src.experiment_tracker.context_update_funcs as context_update_funcs
import src.reporting.reporter_funcs as reporter_funcs
import src.experiment_tracker.context_update_funcs as context_update_funcs
import src.shipment.shipment_funcs as shipment_funcs
import src.reporting.viz_funcs as viz_funcs

MODULE_NAMES_DICT = {
    "analysis": analysis_funcs,
    "data_source": data_source_funcs,
    "quality_assurance": quality_assurance_funcs,
    "quality_control": quality_control_funcs,
    "evaluation": evaluate_funcs,
    "bayes_opt": context_update_funcs,
    "reporter": reporter_funcs,
    "visualization": viz_funcs,
    "experiment_tracker": context_update_funcs,
    "shipment": shipment_funcs,
}


def load_funcs(config, module_name, return_type="dict"):
    """
    Loads functions from a module and returns them as a dictionary of Prefect tasks.

    Args:
        config (dict): A dictionary of function configurations.
        module_name (str): The name of the module to load functions from.
        return_type (str): The type of return value. Can be "dict", "list", or "handle".
            "dict": Returns a dictionary of function names as keys and function handles as values (handles are wrapped in Prefect tasks).
            "list": Returns a list of function handles (handles are wrapped in Prefect tasks).
            "handle": Returns a handle to a single function (handle is wrapped in Prefect task).
    """
    # Define tasks
    tasks = {}
    # Get the correct module from the dictionary
    module = MODULE_NAMES_DICT[module_name]
    
    for func_name, params in config.items():
        function = getattr(
            module, func_name
        )  # Get function from the correct module

        # Create Prefect tasks
#        @task(name=func_name)
        def wrapper_function(*args, function=function, params=params):
            if params is None:
                params = {}
            return function(*args, **params)

        tasks[func_name] = wrapper_function

    if return_type == "dict":
        return tasks
    elif return_type == "list":
        return list(tasks.values())
    elif return_type == "handle":
        assert len(tasks) == 1, "Only one function is allowed to be returned as a handle"
        return tasks[list(tasks.keys())[0]]
    else:
        raise ValueError(f"Invalid return type: {return_type}")

