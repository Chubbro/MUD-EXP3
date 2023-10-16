import numpy as np

def ducb(N, M, T, env):
    u_hat = np.zeros(N)
    l_cum = 0                   # cumulative real loss.
    l_seq = np.zeros(T)         # record the loss history with time.
    ell_tab = np.zeros((N,T))
    s_cnt = np.zeros((N,T))     # sample counter
    s_cnt[:,0] = 1              # padding 1
    arm_cnt = np.zeros(N)       # arm pulling counter

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
        
    for t in range(N, T):
        u_hat = np.sum(ell_tab[:,:t], axis=1)/np.sum(s_cnt[:,:t], axis=1)
        u_bar = u_hat - np.sqrt(arm_cnt*M/np.sum(s_cnt[:,:t], axis=1))       # C Vernade, 2017, UAI
        a = u_bar.argmin()
        loss, delay = env(a, t)
        for j in range(M):
                if t+delay[j] < T:
                    ell_tab[a][t+delay[j]] += loss[j]
                    s_cnt[a][t+delay[j]] += 1
        arm_cnt[a] += 1
        l_cum += np.sum(loss)
        l_seq[t] = l_cum

    return l_seq

