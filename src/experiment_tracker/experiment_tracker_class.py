from ax.service.utils.instantiation import FixedFeatures
from src.utils import load_funcs
from typing import Dict, Any
import plotly.graph_objects as go
import copy
from ax.plot.render import plot_config_to_html

class ExperimentTracker:
    """
    Class for tracking the experiment context via an Ax client. Usually one experiment per participant.
    """
    def __init__(self, config: Dict[str, Any]):
        self._update_ax_client_function = load_funcs(config["update_ax_client_function"], "experiment_tracker", return_type="handle")
        self.current_participant = None
        self._ax_client = None
        self.first_trial_init_function = load_funcs(config["first_trial_init_function"], "experiment_tracker", return_type="handle")
    

    def update_ax_client(self, participant: str) -> Dict[str, Any]:
        """
        Update the experiment context based on the evaluation result.
        
        Parameters
        ----------  
        participant : str
            The participant ID
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing the updated context
        """
        self.current_participant = participant
        self._ax_client, self._ax_client_path = self._update_ax_client_function(participant)


    def get_trial_index(self) -> int:
        """
        Get the index of the current trial
        """
        putative_index = len(self._ax_client.experiment.trials) - 1
        if self._ax_client.experiment.trials[putative_index].status.is_running:
            return putative_index
        elif len(self._ax_client.experiment.running_trial_indices) > 0:
            print(f"Trial {putative_index} is not running, trying {self._ax_client.experiment.running_trial_indices[0]}")
            putative_index = self._ax_client.experiment.running_trial_indices[0]
            return putative_index
        else:
            print(f"Trial {putative_index} is also not running, forcing new trial {putative_index + 1}")
            return putative_index + 1
    

    def update_experiment(self, evaluation_result: Dict[str, float]) -> Dict[str, Any]:
        """
        Complete current trial with results and get parameters for next trial.
        
        Parameters
        ----------
        evaluation_result : Dict[str, float]
            Dictionary containing the evaluation metrics
            
        Returns
        -------
        """
        if self._ax_client is None:
            raise RuntimeError("No active Ax client. Call update_context first.")
        
        if len(self._ax_client.generation_strategy.experiment.trials) == 0:
            parameters = self.first_trial_init_function(self.current_participant)
            fixed_features = FixedFeatures(parameters=parameters)
            parameterization, trial_index = self._ax_client.get_next_trial(fixed_features=fixed_features)
            self.current_trial_index = trial_index
        else:
            self.current_trial_index = self.get_trial_index()

            
        
        # Complete the current trial
        self._ax_client.complete_trial(
            trial_index=self.current_trial_index,
            raw_data=evaluation_result
        )
    

    def get_next_trial(self) -> Dict[str, Any]:
        """
        Get parameters for next trial
        """
        # Get parameters for next trial
        parameters, trial_index = self._ax_client.get_next_trial()
        
        return {
            "parameters": parameters,
            "trial_index": trial_index
        }
    

    def get_contour_plot(self) -> go.Figure:
        """
        Get the contour plot of the experiment
        """
        return self._ax_client.get_contour_plot()
    

    def save_experiment_to_json(self) -> None:
        """
        Save the experiment to a file
        """
        self._ax_client.save_to_json_file(filepath=self._ax_client_path)
    
    
    def save_experiment_to_json_file(self, filepath: str) -> None:
        """
        Save the experiment to a file
        """
        self._ax_client.save_to_json_file(filepath=filepath)


    def get_contour_plot_safe(self) -> "go.Figure":
        """
        Get the contour plot of the experiment by deep copying the Ax client and
        triggering model fitting on the copy. This does NOT affect the real experiment.
        """
        if self._ax_client is None:
            raise RuntimeError("No active Ax client. Call update_context first.")

        viz_ax_client = copy.deepcopy(self._ax_client)
        viz_ax_client.get_next_trial()  # Triggers GP fit on the copy
        # Convert to plotly figure and return
        return go.Figure(viz_ax_client.get_contour_plot().data)
        
    
    