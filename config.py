import numpy as np

# Environment
SPOT          = 100
STRIKE        = 100
MATURITY      = 21 * 3 / 250   # 63 trading days
VOL           = 0.2
MU            = 0.05
DT            = 1 / 250
KAPPA         = 0.01
C             = 1.5
INIT_POSITION = 0
R             = 0

# RL Hyperparameters
TAU        = 5e-4
GAMMA      = 0.9995
LEARN_RATE = 1e-4

# Network architecture 
STATE_DIM  = 3
ACTION_DIM = 1
HIDDEN_DIM = 64
BATCH_SIZE = 64

# DQN action space 
ACTIONS_LIST     = np.linspace(0, 1, 11)
ACTION_DIMENSION = len(ACTIONS_LIST)

# Training
EPISODES            = 10
SCORE_WINDOW_LENGTH = 200
STOP_AVG_REWARD     = 0

# Output
PLOT = True
REPORT = True