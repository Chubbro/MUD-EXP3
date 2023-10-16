#%%
import numpy as np
# %%
def amud(N, M, T, env):
    big_N = np.arange(N)
    ell_tab = np.zeros((N,T))   
    l_cum = 0                
    l_seq = np.zeros(T)        
    L_hat = np.zeros(N)
    prob = np.full(N, 1/N)
    epoch = 0
    phi = np.zeros(T)
    v_cum = 0

    try:
        for t in range(T):
            a = np.random.choice(big_N, p=prob)
            loss, delay = env(a, t)
            for j in range(M):
                if t+delay[j] < T:
                    ell_tab[a][t+delay[j]] += loss[j]/prob[a]
                    phi[t+delay[j]] += 1
            L_hat += ell_tab[:,t]

            v = M*t - np.sum(phi[:t+1])
            v_cum += v
            if v_cum >= np.power(2,epoch)*M:
                epoch += 1
                L_hat = np.zeros(N)
            
            eta = (1/M)*np.sqrt(np.log(N)/np.power(2,epoch))
            L_hat += ell_tab[:,t]
            W = np.exp(-eta*L_hat)
            W_sum = np.sum(W)
            prob = W/W_sum

            l_cum += np.sum(loss)
            l_seq[t] = l_cum
            
    except Exception as e:
        print("Error:", e)
        print("t=",t,"W=",W,"prob=",prob)
    
    return l_seq
#%%
