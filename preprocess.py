import numpy as np

from scipy.misc import imresize


class ToGrayScale(object):
    def __call__(self, observation):
        return np.dot(observation, [0.299, 0.587, 0.114])


class Resize(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, observation):
        return imresize(observation, self.shape)
