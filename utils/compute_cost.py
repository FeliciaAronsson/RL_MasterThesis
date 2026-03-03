import numpy as np
from utils.bs import bs_price

def compute_cost(policy, n_trails, n_steps, spot, strike, maturity, rate, exp_vol, init_pos, dT, mu, kappa):

    np.random.seed(0)

    # Simulera
    sim_paths = np.zeros((n_steps + 1, n_trails))
    sim_times = np.linspace(0, maturity, n_steps + 1)

    sim_paths[0, :] = spot

    # GBM
    for t in range(n_steps):
        Z = np.random.randn(n_trails)
        sim_paths[t+1,:] = sim_paths[t,:] * np.exp((mu - 0.5 * exp_vol**2) * dT + exp_vol * np.sqrt(dT) * Z)
    rew = np.zeros((n_steps, n_trails))

    pos_prev = init_pos * np.ones(n_trails)
    maturity = maturity * np.ones(n_trails)

    pos_next = policy(sim_paths[0,:]/strike, maturity, pos_prev)

    # Hedging loop 
    for timeidx in range(1, n_steps + 1):

        T_prev = maturity - sim_times[timeidx - 1]
        T_next = np.maximum(0, maturity - sim_times[timeidx])


        rew[timeidx - 1, :] = ((sim_paths[timeidx, :] - sim_paths[timeidx - 1, :]) * pos_prev  # ev pos_next(?)
            - np.abs(pos_next - pos_prev) * sim_paths[timeidx - 1,:] * kappa
            - bs_price(sim_paths[timeidx, :], strike, rate, T_next, exp_vol) 
            + bs_price(sim_paths[timeidx - 1, :], strike, rate, T_prev, exp_vol))
        
        if timeidx == n_steps: 
            # Avveckla hedgen
            rew[timeidx - 1, :] -= pos_next * sim_paths[timeidx, :] * kappa
        else:
            pos_prev = pos_next
            pos_next = policy(sim_paths[timeidx,:]/strike, T_next, pos_prev) # t_next ist för t_prev
            
    perCost = np.sum(rew, axis = 0)

    return perCost