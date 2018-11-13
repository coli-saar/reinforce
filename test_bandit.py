import unittest
import numpy as np
from bandit import make_bandit


class TestBandit(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(42)
        self.bandit = make_bandit(10, rng=rng)

    def test10Armed(self):
        res = [[self.bandit.pull(i) for j in range(10000)] for i in range(10)]
        #print(["bandit expected value: %f empirical mean: %f" % (self.bandit.values[i], np.mean(res[i])) for i in range(10)])
        assert all([abs(expected - mean) <= 0.01] for (expected, mean) in zip(self.bandit.values, res))
