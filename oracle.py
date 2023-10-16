import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from env import env_stc, env_nonstc, env_any
from mud import mud
from amud import amud
from ducb import ducb
from se import se
from rand import rand

def oracle(T, env, a):
    l_cum = 0              
    l_seq = np.zeros(T) 
    
    for t in range(T):
        loss, _ = env(a, t)
        l_cum += np.sum(loss)
        l_seq[t] = l_cum
    
    return l_seq