import numpy as np

from gym import ObservationWrapper, RewardWrapper


class TransformObservationWrapper(ObservationWrapper):
    def __init__(self, env, transform=None):
        super(TransformObservationWrapper, self).__init__(env)
        self.transform = transform

    def _observation(self, observation):
        if self.transform:
            return self.transform(observation)


class ScaleRewardWrapper(RewardWrapper):
    def _reward(self, reward):
        return np.clip(reward, -1.0, 1.0)
