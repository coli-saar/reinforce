import numpy as np


class Bandit:

    def __init__(self, mean_values):
        """

        :param mean_values: list of expected values, one for each arm of the bandit (each is a gaussian with this expected value and variance 1).
        """

        self.n_arms = len(mean_values)
        self.values = [value for value in mean_values]

    def pull(self, lever):
        """
        pull lever and get a reward from a normal distribution centered around the lever's expected value
        :param lever: lever number (int)
        :return: reward (float)
        """
        return np.random.normal(self.values[lever], 1., 1)

    def seed(self, seed_value):
        """
        set the seed
        :param seed_value:
        :return:
        """
        np.random.seed(seed_value)


def make_bandit(n_arms, mean_value=0):
    return Bandit(np.random.normal(mean_value, 1., n_arms))

