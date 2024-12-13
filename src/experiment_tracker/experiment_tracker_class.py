

class ExperimentTracker:
    def __init__(self, config):
        self.config = config
        self.experiments = {}  # Dict to store experiment metadata
        self.logger = get_run_logger()
        
    # def get_or_create_experiment(self, experiment_id):
    #     """Get or create an experiment instance"""
    #     if experiment_id not in self.experiments:
    #         self.experiments[experiment_id] = {
    #             'participants': {},
    #             'metadata': {},
    #             'creation_date': datetime.now(),
    #             'status': 'active'
    #         }
    #     return self.experiments[experiment_id]
    
    # def add_session(self, experiment_id, participant_id, session_data):
    #     """Add a new session to the experiment/participant history"""
    #     experiment = self.get_or_create_experiment(experiment_id)
    #     if participant_id not in experiment['participants']:
    #         experiment['participants'][participant_id] = {
    #             'sessions': [],
    #             'current_parameters': None,
    #             'optimization_history': [],
    #             'status': 'active'
    #         }
        
    #     experiment['participants'][participant_id]['sessions'].append(session_data)
        
    # def get_optimization_state(self, experiment_id, participant_id):
    #     """Get current optimization state for a participant"""
    #     return self.experiments[experiment_id]['participants'][participant_id]['optimization_history']