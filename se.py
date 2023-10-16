import numpy as np
# [2021 ICML] Stochastic Multi-Armed Bandits with Unrestricted Delay Distributions

def se(N, M, T, env):
    u_hat = np.zeros(N)
    l_cum = 0    
    l_seq = np.zeros(T)         
    ell_tab = np.zeros((N,T))
    s_cnt = np.zeros((N,T))    
    s_cnt[:,0] = 1             
    arm_cnt = np.zeros(N)       
    S = list(range(N))

    for t in range(N):
        a = t
        loss, delay = env(a, t)
        for j in range(M):
                if t+delay[j] < T:
                    ell_tab[a][t+delay[j]] += loss[j]
                    s_cnt[a][t+delay[j]] += 1
        arm_cnt[a] += 1
        l_cum += np.sum(loss)
        l_seq[t] = l_cum
    
    t = N
    #print("N =", N, "t =", t, "S =", S)
    
    while t < T:
        u_hat = np.sum(ell_tab[:,:t], axis=1)/np.sum(s_cnt[:,:t], axis=1)
        u_bar_low = u_hat - np.sqrt(2*np.log(T)/np.sum(s_cnt[:,:t], axis=1))
        u_bar_up = u_hat + np.sqrt(2*np.log(T)/np.sum(s_cnt[:,:t], axis=1))

        S_ = S
        for i in S_:
            if u_bar_low[i] >= np.min(u_bar_up[S_]):
                S.remove(i)

        for a in S:
            loss, delay = env(a, t)
            for j in range(M):
                if t+delay[j] < T:
                    ell_tab[a][t+delay[j]] += loss[j]
                    s_cnt[a][t+delay[j]] += 1
            arm_cnt[a] += 1
            l_cum += np.sum(loss)
            l_seq[t] = l_cum
            t += 1
            if t == T:
                break

    return l_seq

#%%
