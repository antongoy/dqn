import gym
import torch
import argparse
import numpy as np

from torch.autograd import Variable
from torchvision.transforms import Compose

from models import DQN
from decay import ExpDecay
from replay import ExperienceReplay
from agent import Agent, EpsilonGreedyStrategy, QLearningStrategy, RandomStrategy, NoLearnStrategy
from preprocess import ToGrayScale, Resize


def evaluate_model(Q, states):
    states = torch.cuda.FloatTensor(states)
    states = Variable(states, requires_grad=False)

    values, _ = Q(states).max(dim=1)

    return values.mean().data.cpu().numpy()


parser = argparse.ArgumentParser(description="Learn playing in Space Invaders using DQN")
parser.add_argument('--num-frames', type=int, default=4, help='Number of observed frames for a state')
parser.add_argument('--num-episodes', type=int, default=100, help='Number of episodes to play')
parser.add_argument('--start-decay', type=float, default=0.9, help='Value to start decaying')
parser.add_argument('--decay-factor', type=float, default=0.999, help='Decay factor')
parser.add_argument('--min-decay', type=float, default=0.1, help='Minimum achievable decay value')
parser.add_argument('--replay-size', type=int, default=10000, help='Experience replay size')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
parser.add_argument('--gamma', type=float, default=0.8, help='Discount factor')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--skip-frames', type=int, default=3, help='Skip number of frames')
parser.add_argument('--frame-size', type=int, default=84, help='Frame size')
parser.add_argument('--save-every', type=int, default=20, help='Save every N episodes')

args = parser.parse_args()

env = gym.make('SpaceInvaders-v0')

preprocess = Compose([
    ToGrayScale(),
    Resize((args.frame_size, args.frame_size))
])

Q = DQN(args.num_frames, env.action_space.n).cuda()
decay = ExpDecay(args.start_decay, args.decay_factor, args.min_decay)
test_replay = ExperienceReplay(1000, args.num_frames, (args.frame_size, args.frame_size))
replay = ExperienceReplay(args.replay_size, args.num_frames, (args.frame_size, args.frame_size))

random_agent = Agent(
    RandomStrategy(env.action_space.n),
    NoLearnStrategy()
)

agent = Agent(
    EpsilonGreedyStrategy(Q, env.action_space.n, decay),
    QLearningStrategy(Q, args.lr, replay, args.gamma, args.batch_size)
)

for episode in range(10):

    observation = env.reset()
    observation = preprocess(observation)

    state = test_replay.register_initial_observation(observation)

    done = False
    while not done:
        action = random_agent.select_action(state)
        observation, reward, done, _ = env.step(action)
        reward = np.clip(reward, -1., 1.)
        observation = preprocess(observation)
        state = test_replay.register_transition(observation, action, reward)

test_states, _, _, _ = test_replay.sample_minibatch(4 * args.batch_size)

for episode in range(args.num_episodes):
    observation = env.reset()
    observation = preprocess(observation)

    state = replay.register_initial_observation(observation)

    done = False
    total_reward = 0
    frame = 0

    while not done:
        frame += 1
        # env.render()

        action = agent.select_action(state)

        observation, reward, done, _ = env.step(action)
        observation = preprocess(observation)
        reward = np.clip(reward, -1., 1.)
        state = replay.register_transition(observation, action, reward)

        total_reward += reward

        if frame % args.skip_frames == 0:
            continue

        agent.learn()

    model_score = evaluate_model(Q, test_states)

    print('Episode = {}, Total reward = {}, Decay = {}, Replay = {} Score = {:.3f}'.\
          format(episode, total_reward, decay, replay.counter, model_score[0]))

    if episode and episode % args.save_every == 0:
        torch.save(Q, '../model_after_{}.pth'.format(episode))

torch.save(Q, '../model_final.pth')
