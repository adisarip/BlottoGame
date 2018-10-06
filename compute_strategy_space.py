
from functools import reduce
import operator as op


def compute_strategy_space_size(n_soldiers, n_battlefields):
    n = n_soldiers + n_battlefields - 1
    k = n_battlefields - 1
    r = min(k, n - k)
    N = reduce(op.mul, range(n, n - r, -1), 1)
    D = reduce(op.mul, range(1, r + 1), 1)
    return N // D


print("Input the Number of Soldiers")
inp_n_soldiers = input("[User]: ")
print("Input the Number of Battlefronts")
inp_n_battlefields = input("[User]: ")

n_soldiers = int(inp_n_soldiers)
n_battlefields = int(inp_n_battlefields)
# Instantiate Game and Bot
print("Computing the Strategy Space ...\n")
value = compute_strategy_space_size(n_soldiers, n_battlefields)
print("Strategy Space for {} Soldiers and {} Battlefields is: {}\n".format(n_soldiers, n_battlefields, value))
