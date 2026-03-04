import numpy as np 

class OUNoice():
    def __init__(self, mu, sigma, theta, dT, x0 ):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dT = dT
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = ( self.x_prev + self.theta *(self.mu - self.x_prev) 
            + self.sigma * np.sqrt(self.dT) * np.random.normal(size = self.mu.shape)
            )
        self.x_prev = x
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)