from bandit import make_bandit, Bandit
import numpy as np

bandit = make_bandit(10)
res = [[bandit.pull(i) for j in range(10000)] for i in range(10)]

for i in range(10):
    print("bandit expected value: %f empirical mean: %f" % (bandit.values[i], np.mean(res[i])))


