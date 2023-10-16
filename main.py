import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from env import env_stc, env_nonstc, env_any
from mud import mud
from amud import amud
from ducb import ducb
from se import se
from rand import rand
from oracle import oracle
from plotter import line_plt, fb_plt


def main():
    np.random.seed(2)
    N = 10
    M = 10
    T = 80001
    dm = 10
    num = 3
    #env_st = env_stc(N, M)
    env_ad = env_nonstc(N, M, T, dm)
    #env_ad = env_any(N, M, T, dm, num)
    rep = 10
    l_mud = np.zeros([rep, T])
    l_ducb = np.zeros([rep, T])
    l_amud = np.zeros([rep, T])
    l_rand = np.zeros([rep, T])
    l_se = np.zeros([rep, T])
    l_oracle = np.zeros([rep, T])

    np.random.seed()

    for i in range(rep):
        l_mud[i] = mud(N, M, dm, T, env_ad)
        l_amud[i] = amud(N, M, T, env_ad)
        l_ducb[i] = ducb(N, M, T, env_ad)
        l_se[i] = se(N, M, T, env_ad)
        l_rand[i] = rand(N, T, env_ad)
        l_oracle[i] = oracle(T,env_ad, 1)
        
    data = [(l_mud,"mud"), (l_amud,"amud"), (l_ducb,"ducb"), (l_se,"se"), (l_rand,"random"), (l_oracle,"oracle")]

    fb_plt(data)



if __name__ == '__main__':
    main()

