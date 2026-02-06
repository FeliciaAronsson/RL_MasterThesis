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

    def step(self):
        moneyness = self.spot / self.strike
        timeToMaturity = self.maturity
        position = self.initPosition

        #GBM
        spot_next = self.spot * ((1 + self.mu * self.dT) + (np.random.randn() * self.vol) * np.sqrt(self.dT))
        timeToMaturity_next = max(0, self.maturity - self.dT)

        

        



        return timeToMaturity_next
    
    def reset(self):
        restartState = np.array([self.spot / self.strike, self.maturity, self.initPosition])
        return restartState