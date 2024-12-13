import importlib
import yaml
from prefect import task

# from configs_and_globals.configs import analysis_config
import src.analysis.analysis_funcs as analysis_funcs
import src.analysis.aggregation_criteria as aggregation_criteria
import polars as pl


def load_analysis_funcs(config):
    # Define tasks
    tasks = {}
    for func_name, params in config.items():
        function = getattr(
            analysis_funcs, func_name
        )  # Get function from analysis_funcs module

        # Create Prefect tasks
        @task(name=func_name)
        def wrapper_function(data, function=function, params=params):
            return function(data, **params)

        tasks[func_name] = wrapper_function

    return tasks


def run_analyses(data, tasks):
    # Run reach task sequentially. Save the output of each analysis in a dictionary
    analyses = {}
    for name, task_function in tasks.items():
        analyses[name] = task_function(data)
    return analyses


class AnalysisPipe:
    """
    A class to run analyses on a set of sessions.
    Also handles aggregation criteria and tags.
    """

    def __init__(self, analysis_config):
        self.config_dict = analysis_config
        self.tasks = load_analysis_funcs(self.config_dict["functions"])
        # self.aggregation_criteria_func = load_aggregation_criteria(self.config_dict['aggregation_criteria'])
        self._setup_aggregation()

    def _setup_aggregation(self):
        if agg_criteria := self.config_dict["aggregation_criteria"]:
            self.agg_func_name = list(agg_criteria["agg_func"].keys())[0]
            kwargs = agg_criteria["agg_func"][self.agg_func_name]
            if self.agg_func_name:
                # Aggregation criteria function takes in a polars table of sessions and a list of raw data (polars DataFrames) corresponding to each session.
                # See src.analysis.aggregation_criteria for specific functions.
                self.aggregation_criteria_func = getattr(
                    aggregation_criteria, self.agg_func_name
                )
            else:
                raise ValueError(
                    "'function_name' key must be specified in aggregation_criteria"
                )
            if agg_criteria["buffer_sessions"]:
                self.include_time_buffers = True
                self.buffer_step_size = agg_criteria["buffer_sessions"]["step_size"]
        else:
            self.aggregation_criteria_func = None

    def run_analysis(self, data: pl.DataFrame):
        # Run analyses
        return run_analyses(data, self.tasks)

    def run_aggregation_criteria(self, sessions, data):
        if self.aggregation_criteria_func:
            kwargs = self.config_dict["aggregation_criteria"]["agg_func"][
                self.agg_func_name
            ]
            return self.aggregation_criteria_func(sessions, data, **kwargs)
        else:
            return sessions  # Return original sessions if no aggregation criteria

    def add_time_buffers(self, relevant_data: list[pl.DataFrame]):
        # Include null between sessions to prevent edge effects
        relevant_data_new = []
        for i in range(len(relevant_data) - 1):
            relevant_data_new.append(relevant_data[i])
            # Get the end time of the current session and the start time of the next session
            start_time = relevant_data[i].get_column("localTime").max()
            end_time = relevant_data[i + 1].get_column("localTime").min()
            # Create a new dataframe with a time range between the end of the current session and the start of the next session.
            # Use the buffer step size from the config to set the interval for each row. Should be determined by sample rate.
            relevant_data_new.append(
                pl.DataFrame(
                    {
                        "localTime": pl.datetime_range(
                            start=start_time,
                            end=end_time,
                            interval=self.buffer_step_size,
                            closed="none",
                            eager=True,
                        )
                    }
                )
            )
        relevant_data_new.append(relevant_data[-1])
        return relevant_data_new

    def update_data_with_aggregation_criteria(self, sessions_info, sessions_data):
        sessions_data_grouped = []
        # Split the sessions info dataframe into groups
        sessions_grouped = sessions_info.with_row_count("SessionIndex").partition_by(
            "Group"
        )

        # For each group, get the relevant data from sessions_data and concatenate it into a single dataframe
        for sessions_group in sessions_grouped:
            relevant_data = [
                sessions_data[i]
                for i in sessions_group.get_column("SessionIndex").to_list()
            ]
            if self.include_time_buffers:
                relevant_data = self.add_time_buffers(relevant_data)
            sessions_data_grouped.append(
                pl.concat(relevant_data, how="diagonal").sort("localTime")
            )

        return sessions_grouped, sessions_data_grouped
