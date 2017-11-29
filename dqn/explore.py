import os
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .history import History, StateGenerator
from .transforms import ToGrayScale, Resize
from .epsilon import Epsilon
from .models import DQN

from copy import deepcopy
from random import random
from PIL import Image

from torch.autograd import Variable
from torchvision.transforms import Compose


def main():
    transform = Compose([
        ToGrayScale(),
        Resize((32, 32))
    ])

    batch_size = 32
    env = gym.make('SpaceInvaders-v0')
    dqn = DQN(4, env.action_space.n)
    dqn_ = deepcopy(dqn)
    history = History((32, 32), 4, 100000)
    state_gen = StateGenerator((32, 32), 4)

    epsilon = Epsilon()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(dqn.parameters(), lr=0.00001)

    dqn.train(True)
    dqn_.train(False)

    frame = 0
    running_reward = 0

    for episode in range(1000):
        done = False
        episode_frame = 0
        episode_reward = 0

        observation = env.reset()
        observation = transform(observation)
        state_gen.reset()
        state = state_gen.produce_state(observation)
        history.register_init_state(state)

        while not done:
            eps_value = epsilon(frame)

            if random() < eps_value:
                action = np.random.choice(env.action_space.n, size=1)
            else:
                state = Variable(torch.from_numpy(state).float(), requires_grad=False)
                values = dqn(state)
                _, action = values.max(dim=1, keepdim=True)
                action = action.data.cpu().numpy()

            observation, reward, done, _ = env.step(action)
            observation = transform(observation)
            state = state_gen.produce_state(observation)
            history.register_transition(action, reward, state)

            if frame and frame % 3:
                if frame > batch_size:
                    prev_states, actions, rewards, states, isterminals = history.batch(batch_size)

                    prev_states = Variable(torch.from_numpy(prev_states).float(), requires_grad=False)
                    actions = Variable(torch.from_numpy(actions).long(), requires_grad=False)
                    rewards = Variable(torch.from_numpy(rewards).float(), requires_grad=False)
                    states = Variable(torch.from_numpy(states).float(), requires_grad=False)
                    isterminals = Variable(torch.from_numpy(isterminals).float(), requires_grad=False)

                    values = dqn(prev_states).gather(dim=1, index=actions)
                    target_values, _ = dqn_(states).max(dim=1, keepdim=True)
                    target_values = rewards + 0.99 * (1 - isterminals) * target_values

                    loss = criterion(values, target_values.detach())
                    # loss = torch.clamp(loss, -1, 1)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if frame and not frame % 10000:
                dqn_ = deepcopy(dqn)
                dqn.train(True)
                dqn_.train(False)

            frame += 1
            episode_frame += 1
            episode_reward += reward

        if running_reward:
            running_reward = 0.9 * running_reward + 0.1 * episode_reward
        else:
            running_reward = episode_reward

        if episode and not episode % 10:
            torch.save(dqn, os.path.join('./checkpoints/', 'after_episode_%d.checkpoint' % episode))

        print('Episode {:3}: reward = {:3}, running_reward = {:3.2f} frames = {:5}, total_frames = {:6}' \
              .format(episode, episode_reward, running_reward, episode_frame, frame))

        if episode and not episode % 10:
            done = False
            observation = env.reset()
            observation = transform(observation)
            state_gen.reset()
            state = state_gen.produce_state(observation)
            history.register_init_state(state)
            eval_reward = 0
            while not done:
                # env.render()
                if random() < 0.05:
                    action = np.random.choice(env.action_space.n, size=1)
                else:
                    state = Variable(torch.from_numpy(state).float(), requires_grad=False)
                    values = dqn_(state)
                    _, action = values.max(dim=1, keepdim=True)
                    action = action.data.cpu().numpy()

                observation, reward, done, _ = env.step(action)
                observation = transform(observation)
                state = state_gen.produce_state(observation)
                history.register_transition(action, reward, state)
                eval_reward += reward

            print('Eval episode: reward = {:3}'.format(eval_reward))


if __name__ == '__main__':
    main()