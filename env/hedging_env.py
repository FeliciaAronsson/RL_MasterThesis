import numpy as np
from utils.bs import bs_price

class HedgingEnv:
    def __init__(self, spot, strike, maturity, vol, mu, dT, kappa, c, init_position, r):
        """
        Docstring for __init__
        
        :param self: Description
        :param spot: The current market price ogf the underlying asset
        :param strike: The set price that an option can be excercised 
        :param maturity: Time remaining until the option can be excercised. Acts like a horizion for each episode.
        :param vol: Expected volatility, the riskiness of the stock price. Higher vol makes hedging tasks much harder because the price mves more unpredictable
        :param mu: Description
        :param dT: Time step, the frequency of trading, how many steps that are in each episode
        :param kappa: Risk Aversion parameter, "how much should the agent dislike variance"
        :param c: Transaction cost, the cost of trading, a fee for every dollar of stoock thats been bought or sold. 
        :param init_position: Description
        :param r: Description
        """
       # Value used to fulle reset the enviroment for next episode
        self.start_spot = spot
        self.start_maturity = maturity
        self.start_position = init_position

        # värden som ska ändras under step
        self.spot = spot
        self.strike = strike
        self.maturity = maturity
        self.vol = vol
        self.mu = mu
        self.dT = dT
        self.kappa = kappa
        self.c = c
        self.initPosition = init_position
        self.rate = r

        self.obs_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.obs_high = np.array([10.0, maturity, 1.0], dtype=np.float32)

        self.action_low = 0.0
        self.action_high = 1.0


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
                        - abs((action - pos_prev) * spot_prev) * self.kappa 
                        - bs_price(spot_next, self.strike, self.rate, ttm_next, self.vol) 
                        + bs_price(spot_prev, self.strike, self.rate, ttm_prev, self.vol))

        #step_reward = ((spot_next - spot_prev) * self.initPosition 
         #       - abs((action - pos_prev) * spot_next) * self.kappa 
          #      - bs_price(spot_next, self.strike, self.rate, ttm_next, self.vol) 
           #     + bs_price(spot_prev, self.strike, self.rate, ttm_prev, self.vol))
        
        if done: 
            step_reward -= action * spot_next * self.kappa
           
        
        # Pentalty
        reward = step_reward - self.c * step_reward**2

        state_next = np.array([spot_next / self.strike, ttm_next, action])

        # Uppdatera miljöns state
        self.maturity = ttm_next
        self.initPosition = action
        self.spot = spot_next

        return reward, state_next, done
    
    def reset(self):
        self.spot = self.start_spot
        self.maturity = self.start_maturity
        self.initPosition = self.start_position

        
        state_initial = np.array([self.spot / self.strike, self.maturity, self.initPosition])
        return state_initial