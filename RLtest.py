# Option parameters
Strike = 100
Maturity = 21*3/250

# Asset parameters
SpotPrice = 100
ExpVol = 0.2
ExpReturn = 0.05

# Simulation parameters
rfRate = 0
dT = 1/250
nSteps = Maturity/dT
nTrials = 5000

# Transacation cost and cost function parameters
c = 1.5
kappa = .01
InitPosition = 0

# Set the random generator seed for reproducibility.
rng(3)