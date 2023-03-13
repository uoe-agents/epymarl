from .ppo_learner import PPOLearner

class SAFLearner(PPOLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
