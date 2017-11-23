import numpy as np

from random import random


class Policy(object):
    def __init__(self, env):
        self.env = env

    def decision(self, state):
        raise NotImplementedError


class GreedyPolicy(Policy):
    def __init__(self, env, dqn):
        super(GreedyPolicy, self).__init__(env)

        self.dqn = dqn

    def decision(self, state):
        return self.dqn.argmax(state)


class RandomPolicy(Policy):
    def __init__(self, env):
        super(RandomPolicy, self).__init__(env)

    def decision(self, state):
        return np.random.choice(self.env.action_space.n, size=1)


class EpsilonGreedyPolicy(Policy):
    def __init__(self, env, dqn, epsilon):
        super(EpsilonGreedyPolicy, self).__init__(env)

        self.epsilon = epsilon
        self.random_policy = RandomPolicy(env)
        self.greedy_policy = GreedyPolicy(env, dqn)

    def decision(self, state):
        epsilon = self.epsilon(self.env.frame)

        if random() < epsilon:
            return self.random_policy.decision(state)
        else:
            return self.greedy_policy.decision(state)
