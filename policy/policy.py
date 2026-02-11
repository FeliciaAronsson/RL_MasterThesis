import numpy as np 
from utils.bs import bs_delta

eps = np.finfo(float).eps
K = 1.0

def policy_BSM(mR, ttm, pos, rfRate, exp_vol):
    T = np.maximum(ttm, eps)
    return bs_delta(mR, K, rfRate, T, exp_vol)

