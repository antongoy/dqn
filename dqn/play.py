import gym
import torch
import argparse

from .policy import GreedyPolicy
from .environment import TransformObservationWrapper, ScaleRewardWrapper, BookkeepingWrapper
from .transforms import ToGrayScale, Resize
from .history import History
from .models import WrapperDQN


from torchvision.transforms import Compose


def run_episode(env, policy, history):
    done = False

    observation = env.reset()
    state = history.register_init_observation(observation)
    while not done:
        env.render()
        action = policy.decision(state)
        observation, reward, done, _ = env.step(action)
        state = history.register_transition(observation, action, reward)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Total number of episodes to learn')
    parser.add_argument('--use-cuda', action='store_true', help='Use CUDA')

    args = parser.parse_args()

    transform = Compose([
        ToGrayScale(),
        Resize((84, 84))
    ])

    dqn = torch.load(args.model_path, map_location={'cuda:0': 'cpu'})
    dqn = WrapperDQN(dqn, cuda=args.use_cuda)

    env = gym.make('SpaceInvaders-v0')
    env = TransformObservationWrapper(ScaleRewardWrapper(env), transform)
    env = BookkeepingWrapper(env, dqn)

    policy = GreedyPolicy(env, dqn)

    history = History(4, 4, (84, 84))

    run_episode(env, policy, history)


if __name__ == '__main__':
    main()