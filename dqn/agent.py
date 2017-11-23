class Agent(object):
    def __init__(self, policy, learning_strategy):
        self.policy = policy
        self.learning_strategy = learning_strategy

    def decision(self, state, *args, **kwargs):
        return self.policy.decision(state, *args, **kwargs)

    def learn(self, states, actions, rewards, next_states):
        return self.learning_strategy.learn(states, actions, rewards, next_states)
