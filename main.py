from bandit import make_bandit, Bandit
import numpy as np
from learner import Learner, plot_reward, print_estimates

n_arms = 10
n_bandits = 2000
n_steps = 1000

bandits = [make_bandit(n_arms) for _ in range(n_bandits)]

greeders = [Learner(n_arms, 5) for _ in range(n_bandits)]


results = [greeders[i].train(n_steps, bandits[i], 0.1) for i in range(n_bandits)]

averages = [np.mean([results[i][j][1] for i in range(n_bandits)]) for j in range(n_steps)]


plot_reward(averages)


#print("Bandit : ", bandit.best_lever,  bandit.values)
