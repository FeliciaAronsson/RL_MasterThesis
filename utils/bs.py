import numpy as np
from scipy.stats import norm 

def bs_price(S, K, r, T, sigma):
    S = np.asarray(S)
    T = np.asarray(T)

    T_safe = np.maximum(T, 1e-12)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_safe) / (sigma * np.sqrt(T_safe))
    d2 = d1 - sigma * np.sqrt(T_safe)

    price = S * norm.cdf(d1) - K * np.exp(-r * T_safe) * norm.cdf(d2)

    payoff = np.maximum(S-K, 0.0)

    price = np.where(T<=0, payoff, price)
    return price 


def bs_price2(S, K, r, T, sigma):
    if T <= 0:
        return np.maximum(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_delta2(S, K, r, T, sigma):
    if T <= 0:   
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def bs_delta(S, K, r, T, sigma):
    S = np.asarray(S)
    T = np.asarray(T)
    T_safe = np.maximum(T, 1e-12)# so that its not zero


    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_safe) / (sigma * np.sqrt(T_safe))
    delta = norm(d1)

                    # (condition, value if true, value if false)
    delta = np.where(T <= 0, (S > K).astype(float), delta)
    return delta
