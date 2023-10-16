import numpy as np
from scipy.stats import truncnorm

from utils import trunc_norm

def env_stc(N, M, dm):
    low, up = 0, 1
    mu_seq = np.random.uniform(low=0, high=1, size=N)
    sig_seq = np.random.uniform(low=0.1, high=0.5, size=N)
    dist_seq = trunc_norm(N, mu=mu_seq, sig=sig_seq, lower=low, upper=up)
    
    def func(a, t):
        dist = dist_seq[a]
        loss = dist.rvs(size=M)
        delay = np.random.randint(low=1, high=dm, size=M)
        return loss, delay
    
    return func

def env_nonstc(N, M, T, dm):
    low, up = 0, 1
    num = 3     # The total different mean points number.
    del_t = T//num + 1
    mu_mat = np.zeros((N, num))
    sig_mat = np.random.uniform(low=0.1, high=0.2, size=(N,num))
    
    for i in range(N):
        mu_mat[i] = np.random.uniform(low=0, high=1, size=num)
    
    def func(a, t):
        dist = truncnorm((low - mu_mat[a][t//del_t]) / sig_mat[a][t//del_t], (up - mu_mat[a][t//del_t]) / sig_mat[a][t//del_t], loc=mu_mat[a][t//del_t], scale=sig_mat[a][t//del_t])
        loss = dist.rvs(size=M)
        delay = np.random.randint(low=1, high=dm, size=M)
        return loss, delay
    
    return func

def env_any(N, M, T, dm, num):
    low, up = 0, 1
    del_t = T//num + 1
    mu_mat = np.zeros((N, num))
    sig_mat = np.random.uniform(low=0.1, high=0.2, size=(N,num))
    
    for i in range(N):
        mu_mat[i] = np.random.uniform(low=0, high=1, size=num)
    
    def func(a, t):
        dist = truncnorm((low - mu_mat[a][t//del_t]) / sig_mat[a][t//del_t], (up - mu_mat[a][t//del_t]) / sig_mat[a][t//del_t], loc=mu_mat[a][t//del_t], scale=sig_mat[a][t//del_t])
        loss = dist.rvs(size=M)
        delay = np.random.randint(low=1, high=dm, size=M)
        return loss, delay
    
    return func

