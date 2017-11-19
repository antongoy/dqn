import gym
import torch
import argparse

from torchvision.transforms import Compose

from agent import Agent, GreedyStrategy, NoLearnStrategy
from replay import StateGenerator
from preprocess import ToGrayScale, Resize


parser = argparse.ArgumentParser()
parser.add_argument('model_path')
parser.add_argument('--env', default='SpaceInvaders-v0')
parser.add_argument('--num_episodes', type=int, default=1)
parser.add_argument('--frame-size', type=int, default=84, help='Frame size')


args = parser.parse_args()

env = gym.make(args.env)

Q = torch.load(args.model_path)
Q = Q.train(False)

state_gen = StateGenerator(4, (args.frame_size, args.frame_size))

agent = Agent(
    GreedyStrategy(Q),
    NoLearnStrategy()
)

preprocess = Compose([
    ToGrayScale(),
    Resize((args.frame_size, args.frame_size))
])

for episode in range(args.num_episodes):
    observation = env.reset()
    observation = preprocess(observation)
    state_gen.register_observation(observation)

    state = state_gen.produce_state()

    done = False
    total_reward = 0

    while not done:
        env.render()

        action = agent.select_action(state)

        observation, reward, done, _ = env.step(action)
        observation = preprocess(observation)

        state_gen.register_observation(observation)
        state = state_gen.produce_state()

        total_reward += reward

    print('Episode reward = {}'.format(total_reward))
