import numpy as np


class ExpDecay(object):
    def __init__(self, initial_value, decay_factor, stop_value):
        self.initial_value = initial_value
        self.decay_factor = decay_factor
        self.stop_value = stop_value
        self.stop_t = int(np.log(stop_value / initial_value) / np.log(decay_factor))
        self.t = -1

    def value(self, step):
        if step > self.stop_t:
            return self.stop_value

        return self.initial_value * (self.decay_factor ** step)

    @property
    def step(self):
        self.t += 1
        return self.t

    def __call__(self):
        return self.value(self.step)

    def reset(self):
        self.t = 0

    def __str__(self):
        return "{:.3}".format(self.value(self.t))