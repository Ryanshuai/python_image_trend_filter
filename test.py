# Import packages.
import cvxpy as cp
import numpy as np
import scipy

n = 5

up_paradiagonal = np.zeros((n, n))
for i in range(n - 1):
    up_paradiagonal[i][i + 1] = 1
down_paradiagonal = up_paradiagonal.transpose()

print(up_paradiagonal)
print(down_paradiagonal)