from prefect import task
import src.analysis.analysis_funcs as analysis_funcs
import src.data_source.data_source_funcs as data_source_funcs
import src.quality_assurance.quality_assurance_funcs as quality_assurance_funcs
import src.quality_control.quality_control_funcs as quality_control_funcs
import src.reward.reward_funcs as reward_funcs
import src.bayes_opt.bayes_opt_funcs as bayes_opt_funcs
import src.reporting.reporting_funcs as reporting_funcs


MODULE_NAMES_DICT = {
    "analysis": analysis_funcs,
    "data_source": data_source_funcs,
    "quality_assurance": quality_assurance_funcs,
    "quality_control": quality_control_funcs,
    "reward": reward_funcs,
    "bayes_opt": bayes_opt_funcs,
    "reporting": reporting_funcs,
}

def load_funcs(config, module_name):
    # Define tasks
    tasks = {}
    for func_name, params in config.items():
        function = getattr(
            module_name, func_name
        )  # Get function from analysis_funcs module

        # Create Prefect tasks
        @task(name=func_name)
        def wrapper_function(data, function=function, params=params):
            return function(data, **params)

        tasks[func_name] = wrapper_function

    return tasks