import numpy as np

def mud(N, M, dm, T, env): 
    big_N = np.arange(N)
    ell_tab = np.zeros((N,T)) 
    l_cum = 0               
    l_seq = np.zeros(T)   
    L_hat = np.zeros(N)
    prob = np.full(N, 1/N)
    eta = np.sqrt(np.log(N)*10/(M*T*(N*np.e+4*dm))) 

    for t in range(T):
        a = np.random.choice(big_N, p=prob)
        loss, delay = env(a, t)
        for j in range(M):
            if t+delay[j] < T:
                ell_tab[a][t+delay[j]] += loss[j]/prob[a]
        L_hat += ell_tab[:,t]
        W = np.exp(-eta*L_hat)
        W_sum = np.sum(W)
        prob = W/W_sum

        l_cum += np.sum(loss)
        l_seq[t] = l_cum
    
    return l_seq