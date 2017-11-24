class Epsilon(object):
    def __init__(self, initial_value=1.0, final_value=0.1, annealing=500000):
        self.initial_value = initial_value
        self.final_value = final_value
        self.annealing = annealing

        self.factor = (final_value / initial_value) ** (1. / self.annealing)

    def __call__(self, counter):
        value = self.initial_value * (self.factor ** counter)

        if value < self.final_value:
            return self.final_value

        return value
