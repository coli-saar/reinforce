from bandit import make_bandit, Bandit
import numpy as np
from learner import Learner, plot_reward, print_estimates
from matplotlib import pyplot as plt

n_arms = 10
n_bandits = 2000
n_steps = 1000

#n_arms = 2
#n_bandits = 5
#n_steps = 10

bandits = [make_bandit(n_arms) for _ in range(n_bandits)]

for epsilon in [0., 0.1, 0.01]:

    greeders = [Learner(n_arms, 0) for _ in range(n_bandits)]

    results = [greeders[i].train(n_steps, bandits[i], epsilon) for i in range(n_bandits)]

    averages = [np.mean([results[i][j][1] for i in range(n_bandits)]) for j in range(n_steps)]

    bests = [[results[i][j][0] == bandits[i].best_lever for i in range(n_bandits)] for j in range(n_steps)]
    print(bests)
    sums = [[0 for _ in range(n_steps)] for _ in range(n_bandits)]

    for i in range(n_bandits):
        s = 0
        for j in range(n_steps):
            s = s + bests[j][i]
            #print("%d, %s" % (s, bests[j][i]))
            sums[i][j] = s


    #print(sums)
    averaged_percentages = [np.mean([sums[i][j] for i in range(n_bandits)])/(j+1) for j in range(n_steps)]
    #print(len(averaged_percentages))
    #plot_reward(averages)
    plt.plot(range(1, n_steps+1), averaged_percentages, label=("epsilon="+str(epsilon)))

plt.legend(loc=4)
plt.savefig('/Users/antoine/dave_interpretation.png')
plt.show()
#print("Bandit : ", bandit.best_lever,  bandit.values)


for epsilon in [0., 0.1, 0.01]:

    greeders = [Learner(n_arms, 0) for _ in range(n_bandits)]

    results = [greeders[i].train(n_steps, bandits[i], epsilon) for i in range(n_bandits)]

    averages = [np.mean([results[i][j][1] for i in range(n_bandits)]) for j in range(n_steps)]

    bests = [[results[i][j][0] == bandits[i].best_lever for i in range(n_bandits)] for j in range(n_steps)]
    print(bests)
    percentages = []

    for j in range(n_steps):
        s = 0
        for i in range(n_bandits):
            s = s + bests[j][i]
        percentages.append(s/n_bandits)

    plt.plot(range(1, n_steps+1), percentages, label=("epsilon="+str(epsilon)))

plt.legend(loc=4)
plt.savefig('/Users/antoine/other_interpretation.png')
plt.show()
