import gym
import torch

from .policy import GreedyPolicy
from .environment import TransformObservationWrapper, ScaleRewardWrapper, BookkeepingWrapper
from .transforms import ToGrayScale, Resize
from .history import History


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
    transform = Compose([
        ToGrayScale(),
        Resize((84, 84))
    ])

    dqn = torch.load('./checkpoints/after_episode_90.checkpoint')

    env = gym.make('SpaceInvaders-v0')
    env = TransformObservationWrapper(ScaleRewardWrapper(env), transform)
    env = BookkeepingWrapper(env, dqn)

    policy = GreedyPolicy(env, dqn)

    history = History(4, 4, (84, 84))

    run_episode(env, policy, history)


if __name__ == '__main__':
    main()