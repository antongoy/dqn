import gym
import torch
import argparse

from .models import DQN, WrapperDQN
from .policy import EpsilonGreedyPolicy
from .environment import TransformObservationWrapper, ScaleRewardWrapper, BookkeepingWrapper
from .transforms import ToGrayScale, Resize
from .learning import QLearning
from .history import History
from .epsilon import Epsilon

from torchvision.transforms import Compose


def run_episode(env, policy, learning_strategy, history):
    done = False

    observation = env.reset()
    state = history.register_init_observation(observation)
    while not done:
        action = policy.decision(state)
        observation, reward, done, _ = env.step(action)
        state = history.register_transition(observation, action, reward)
        batch = history.batch()
        learning_strategy.learn(*batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', default=1000, type=int, help='Total number of episodes to learn')
    parser.add_argument('--learn-every', default=5, type=int, help='Frequency in frames of learning')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--annealing', default=300000, type=int, help='Number iterations to anneal epsilon')
    parser.add_argument('--history-size', default=100000, type=int, help='Max number of frames to store in history')
    parser.add_argument('--saved-model-path', default='', help='Path to a model file')
    parser.add_argument('--use-cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--checkpoint-dir-path', default='./checkpoints/', help='Checkpoint dir')

    args = parser.parse_args()

    transform = Compose([
        ToGrayScale(),
        Resize((84, 84))
    ])

    env = gym.make('SpaceInvaders-v0')

    if args.saved_model_path:
        dqn = torch.load(args.saved_model_path)
    else:
        dqn = DQN(4, env.action_space.n)

    dqn = WrapperDQN(dqn, cuda=args.use_cuda)

    env = TransformObservationWrapper(ScaleRewardWrapper(env), transform)
    env = BookkeepingWrapper(env, dqn, print_every=1, checkpoint_dir_path=args.checkpoint_dir_path)

    learning_strategy = QLearning(env, dqn, learn_every=args.learn_every, lr=args.lr)
    policy = EpsilonGreedyPolicy(env, dqn, Epsilon(annealing=args.annealing))

    history = History(args.history_size, 4, (84, 84), batch_size=args.batch_size)

    for episode in range(args.episodes):
        run_episode(env, policy, learning_strategy, history)


if __name__ == '__main__':
    main()
