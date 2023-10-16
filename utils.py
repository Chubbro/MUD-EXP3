# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import truncnorm


def trunc_norm(N, mu, sig, lower, upper):
    dist_dic = {}
    for i in range(N): 
        X = truncnorm((lower - mu[i]) / sig[i], (upper - mu[i]) / sig[i], loc=mu[i], scale=sig[i])
        dist_dic[i] = X        
    return dist_dic
