
#from utils.bs import bs_delta
import numpy as np
from scipy.stats import norm 

def bs_delta(S, K, r, T, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

eps = np.finfo(float).eps  # MATLAB eps
K = 1.0

def policy_BSM(mR, TTM, Pos, rfRate, ExpVol):
    T = max(TTM, eps)
    return bs_delta(mR, K, rfRate, T, ExpVol)

mR = 1.05
TTM = 0.2
Pos = 0.0
rfRate = 0.01
ExpVol = 0.2

print(policy_BSM(mR, TTM, Pos, rfRate, ExpVol))
