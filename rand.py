import numpy as np

def rand(N, T, env):
    big_N = np.arange(N)
    l_cum = 0
    l_seq = np.zeros(T)

    for t in range(T):
        a = np.random.choice(big_N)
        loss, _ = env(a, t)
        l_cum += np.sum(loss)
        l_seq[t] = l_cum

    return l_seq