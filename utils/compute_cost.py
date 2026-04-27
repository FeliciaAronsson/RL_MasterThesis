import numpy as np
from utils.bs import bs_price
from config import MARKET_INPACT

def compute_cost(policy, n_trails, n_steps, spot, strike, maturity, rate, exp_vol, init_pos, dT, mu, kappa):

    np.random.seed(0)

    # Simulate
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

    # For likvidering 
    if MARKET_INPACT != 0:
           # For likvidering 
           # Hedging loop 
        for timeidx in range(1, n_steps + 1):
            trade_size = np.abs(pos_next - pos_prev)
            linear_cost = trade_size * sim_paths[timeidx - 1, :] * kappa
            quadratic_cost = MARKET_INPACT * (np.abs(pos_next - pos_prev)**2) *  sim_paths[timeidx - 1, :]

            T_prev = maturity - sim_times[timeidx - 1]
            T_next = np.maximum(0, maturity - sim_times[timeidx])

            rew[timeidx - 1, :] = ((sim_paths[timeidx, :] - sim_paths[timeidx - 1, :]) * pos_prev
                - (linear_cost + quadratic_cost)
                - bs_price(sim_paths[timeidx, :], strike, rate, T_next, exp_vol) 
                + bs_price(sim_paths[timeidx - 1, :], strike, rate, T_prev, exp_vol))
            
            if timeidx == n_steps: 
                # Final step (matuarity)
                rew[timeidx - 1, :] -= pos_next * sim_paths[timeidx, :] * kappa
                
            else:
                pos_prev = pos_next
                pos_next = policy(sim_paths[timeidx,:]/strike, T_next, pos_prev) 
                
    else:
        # Hedging loop 
        for timeidx in range(1, n_steps + 1):

            T_prev = maturity - sim_times[timeidx - 1]
            T_next = np.maximum(0, maturity - sim_times[timeidx])

            rew[timeidx - 1, :] = ((sim_paths[timeidx, :] - sim_paths[timeidx - 1, :]) * pos_prev
                - np.abs(pos_next - pos_prev) * sim_paths[timeidx - 1,:] * kappa
                - bs_price(sim_paths[timeidx, :], strike, rate, T_next, exp_vol) 
                + bs_price(sim_paths[timeidx - 1, :], strike, rate, T_prev, exp_vol))
            
            if timeidx == n_steps: 
                # Final step (matuarity)
                rew[timeidx - 1, :] -= pos_next * sim_paths[timeidx, :] * kappa
                
            else:
                pos_prev = pos_next
                pos_next = policy(sim_paths[timeidx,:]/strike, T_next, pos_prev) 
                
    perCost = np.sum(rew, axis = 0)

    return perCost