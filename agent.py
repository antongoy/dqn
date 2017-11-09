import torch
import numpy as np

from torch.autograd import Variable


class Agent(object):
    def __init__(self, play_strategy, learn_strategy):
        self.play_strategy = play_strategy
        self.learn_strategy = learn_strategy

    def select_action(self, state):
        return self.play_strategy.select_action(state)

    def learn(self):
        return self.learn_strategy.learn()


class GreedyStrategy(object):
    def __init__(self, Q):
        self.Q = Q

    def select_action(self, state):
        state = state[None, ...]
        state = torch.cuda.FloatTensor(state)
        state = Variable(state, requires_grad=False)

        values = self.Q(state)

        _, action = values.max(dim=1)
        return action.data.cpu().numpy()


class RandomStrategy(object):
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def select_action(self, _):
        return np.random.choice(self.num_actions, size=1, replace=False)


class EpsilonGreedyStrategy(object):
    def __init__(self, Q, num_actions, decay):
        self.decay = decay
        self.random_strategy = RandomStrategy(num_actions)
        self.greedy_strategy = GreedyStrategy(Q)

    def select_action(self, state):
        decay_value = self.decay()
        random_value = np.random.random(size=1)

        if random_value < decay_value:
            return self.random_strategy.select_action(state)
        else:
            return self.greedy_strategy.select_action(state)


class NoLearnStrategy(object):
    def learn(self):
        pass


class SimpleLearnStrategy(object):
    def __init__(self, Q, criterion, optimizer, replay, gamma, batch_size):
        self.Q = Q
        self.gamma = gamma
        self.batch_size = batch_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.replay = replay

    def learn(self):
        prev_states, actions, rewards, states = self.replay.sample_minibatch(self.batch_size)

        rewards = torch.cuda.FloatTensor(rewards[:, None])
        rewards = Variable(rewards, requires_grad=False)

        actions = torch.cuda.LongTensor(actions[:, None])
        actions = Variable(actions, requires_grad=False)

        prev_states = torch.cuda.FloatTensor(prev_states)
        prev_states = Variable(prev_states, requires_grad=False)

        states = torch.cuda.FloatTensor(states)
        states = Variable(states, requires_grad=False)

        current_values = self.Q(prev_states)
        current_values = current_values.gather(dim=1, index=actions)

        values, _ = self.Q(states).max(dim=1, keepdim=True)
        target_values = rewards + self.gamma * values
        target_values = target_values.detach()

        loss = self.criterion(current_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()