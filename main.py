from env.hedging_env import HedgingEnv
import numpy as np

variables = {
    "SpotPrice": 100,
    "Strike": 100,
    "Maturity": 21*3/250,
    "vol": 0.2,
    "mu": 0.05,
    "dT": 1/250,
    "kappa": 0.0 ,
    "c": 0,
    "InitPosition": 0,
    "r": 0.0
}

np.random.seed(0)

env = HedgingEnv(100, 100, 21*3/250, 0.2, 0.05, 1/250, 0.0, 0, 0, 0)

#initial_state = env.reset()
#print(initial_state)

testStep = env.step(1)
print(testStep)