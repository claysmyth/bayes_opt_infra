

class Reward:
    def __init__(self, reward_config: DictConfig):
        self.reward_config = reward_config

    def reward_function(self, session: DictConfig):
        pass