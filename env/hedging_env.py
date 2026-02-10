import numpy as np
from utils.bs import bs_price

class HedgingEnv:
    def __init__(self, spot, strike, maturity, vol, mu, dT, kappa, c, initPosition, r):
        self.spot = spot
        self.strike = strike
        self.maturity = maturity
        self.vol = vol
        self.mu = mu
        self.dT = dT
        self.kappa = kappa
        self.c = c
        self.initPosition = initPosition
        self.rate = r
        

    def step(self, action):
        ttm_prev = self.maturity
        pos_prev = self.initPosition

        spot_prev = self.spot

        # GBM
        spot_next = spot_prev * ((1 + self.mu * self.dT) + (np.random.randn() * self.vol) * np.sqrt(self.dT))
        ttm_next = max(0, self.maturity - self.dT)

        done = ttm_next < 1e-8

        # Reward P&L
        step_reward = ((spot_next - spot_prev) * action 
                       - abs((action - pos_prev) * spot_next) * self.kappa 
                       + bs_price(spot_next, self.strike, self.rate, ttm_next, self.vol) 
                       - bs_price(spot_prev, self.strike, self.rate, ttm_prev, self.vol))
        

        
        if done: 
            step_reward -= action * spot_next * self.kappa
        
        reward = step_reward - self.c * step_reward**2

        state = np.array([spot_next / self.strike, ttm_next, action])
        return reward, state
    
    def reset(self):
        restartState = np.array([self.spot / self.strike, self.maturity, self.initPosition])
        return restartState