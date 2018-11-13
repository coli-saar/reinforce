import numpy as np


class Bandit:

    def __init__(self, mean_values, rng: np.random.RandomState = np.random):
        """

        :param mean_values: list of expected values, one for each arm of the bandit (each is a gaussian with this expected value and variance 1).
        :param rng: the random number generator used in determining levers' outcome
        """

        self.n_arms = len(mean_values)
        self.values = [value for value in mean_values]
        self.rng = rng

    def pull(self, lever):
        """
        pull lever and get a reward from a normal distribution centered around the lever's expected value
        :param lever: lever number (int)
        :return: reward (float)
        """
        return self.rng.normal(self.values[lever], 1., 1)

    def seed(self, seed_value):
        """
        set the seed
        :param seed_value:
        :return:
        """
        self.rng.seed(seed_value)


def make_bandit(n_arms, mean_value=0, rng: np.random.RandomState = np.random):
    return Bandit(rng.normal(mean_value, 1., n_arms))

