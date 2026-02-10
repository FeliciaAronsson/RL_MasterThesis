import numpy as np
from utils.bs import bs_price

class ComputeCost:
    def __init__(self, policy, n_trails, n_steps, spot, strike, maturity, rate, exp_vol, init_pos, dT, mu, kappa):
        self.policy = policy
        self.n_trails = n_trails
        self.n_steps = n_steps
        self.spot = spot
        self.strike = strike
        self.maturity = maturity
        self.rate = rate
        self.exp_vol = exp_vol 
        self.init_pos = init_pos
        self.dT = dT
        self.mu = mu
        self.kappa = kappa

    def compute_cost(self, action):
        ttm_prev = self.maturity
        pos_prev = self.init_pos
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

        state_next = np.array([spot_next / self.strike, ttm_next, action])
        return reward, state_next, done

        pass