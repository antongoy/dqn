import os
import torch
import numpy as np

from gym import Wrapper, ObservationWrapper, RewardWrapper


class TransformObservationWrapper(ObservationWrapper):
    def __init__(self, env, transform=None):
        super(TransformObservationWrapper, self).__init__(env)
        self.transform = transform

    def _observation(self, observation):
        if self.transform:
            return self.transform(observation)


class ScaleRewardWrapper(RewardWrapper):
    def _reward(self, reward):
        return int(np.clip(reward, -1.0, 1.0))


class BookkeepingWrapper(Wrapper):
    def __init__(self, env, dqn,  checkpoint_every=10, checkpoint_dir_path='./checkpoints/', print_every=5):
        super(BookkeepingWrapper, self).__init__(env)
        self.dqn = dqn
        self.print_every = print_every
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir_path = checkpoint_dir_path

        self.episode = 0
        self.episode_frame = 0
        self.frame = 0
        self.total_reward = 0
        self.episode_reward = 0

    def _reset(self, **kwargs):
        if self.episode and not self.episode % self.print_every:
            print('Episode {:3}: reward = {:3}, frames = {:5}, total_frames = {:6}'\
                  .format(self.episode, self.episode_reward, self.episode_frame, self.frame))

        if self.episode and not self.episode % self.checkpoint_every:
            if not os.path.exists(self.checkpoint_dir_path):
                os.mkdir(self.checkpoint_dir_path)
            torch.save(self.dqn.model, os.path.join(self.checkpoint_dir_path, 'after_episode_%d.checkpoint' % self.episode))

        self.episode_frame = 1
        self.episode += 1
        self.frame += 1
        self.episode_reward = 0

        return self.env.reset(**kwargs)

    def _step(self, action):
        self.episode_frame += 1
        self.frame += 1
        observation, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        return observation, reward, done, info
