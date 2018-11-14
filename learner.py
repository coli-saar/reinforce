from bandit import Bandit
import numpy as np
from matplotlib import pyplot

class Learner:

    def __init__(self, n_arms, init_estimate=0, rng:np.random.RandomState = np.random):
        '''
        :param n_arms: # levers
        '''

        self.n_arms = n_arms
        # initialize estimated expected reward for every lever
        self.Q = [init_estimate] * self.n_arms
        # initialize count of pulls for every lever
        self.N = [0] * self.n_arms
        # initialize rng, a random generator for the learner
        self.rng = rng

    def step(self, bandit : Bandit, epsilon= 0):
        '''
        action selection through greedy method
        :param bandit:
        :return:
        '''

        #flip coin to decide a greedy action or random
        flip_coin = self.rng.rand()

        if flip_coin < epsilon:
            #random action
            chosen_lever = self.rng.randint(self.n_arms)

        else:
            # find all the best levers and then sample randomly from them
            reward = max(self.Q)
            best_levers = [lever for lever in range(self.n_arms) if self.Q[lever] == reward]
            chosen_lever = best_levers[self.rng.randint(len(best_levers))]

        reward = bandit.pull(chosen_lever)

        self.N[chosen_lever] += 1
        #Updated estimated expected reward for the best lever
        self.Q[chosen_lever] = self.Q[chosen_lever] + (1/self.N[chosen_lever])*(reward - self.Q[chosen_lever])

        return chosen_lever, reward


    def train(self, n_iterations, bandit : Bandit, epsilon=0):
        '''
        train the leaner
        :param n_iterations:
        :return:
        '''

        results = [self.step(bandit, epsilon) for i in range(n_iterations)]

        return results


def print_estimates(learner):
    print("Learner : ", np.argmax(learner.Q), learner.Q)


def plot_reward(rewards):

    pyplot.plot([reward for reward in rewards])

    pyplot.show()



