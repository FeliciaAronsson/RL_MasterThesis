
import numpy as np
from scipy.stats import norm 

def bs_price(S, K, r, T, sigma):
    """
    Calculate option price, Black Scholes model
    
    :param S: Spot price
    :param K: Strike price
    :param r: Risk free rate
    :param T: Time to maturity
    :param sigma: Standard deviation
    """
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    
    #T_safe, so division with zero is not possible
    T_safe = np.maximum(T, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_safe) / (sigma * np.sqrt(T_safe))
    d2 = d1 - sigma * np.sqrt(T_safe)
    
    price = S * norm.cdf(d1) - K * np.exp(-r * T_safe) * norm.cdf(d2)
    return np.where(T <= 1e-8, np.maximum(S - K, 0.0), price)

def bs_delta(S, K, r, T, sigma):
    """
    calculate delta for Black Scholes model.
    
    :param S: Spot price
    :param K: Strike price
    :param r: Risk free rate
    :param T: Time to maturity
    :param sigma: Standard deviation
    """
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    
    T_safe = np.maximum(T, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_safe) / (sigma * np.sqrt(T_safe))
    
    delta = norm.cdf(d1)
    return np.where(T <= 1e-8, np.where(S > K, 1.0, 0.0), delta)