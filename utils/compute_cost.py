import numpy as np
from utils.bs import bs_price
from policy.policy import policy_BSM


def compute_cost(policy, n_trails, n_steps, spot, strike, maturity, rate, exp_vol, init_pos, dT, mu, kappa):
    #ttm_prev = self.maturity
    #pos_prev = self.init_pos
    #spot_prev = self.spot

    sim_paths = np.zeros((n_steps + 1, n_trails))
    sim_times = np.linspace(0, maturity, n_steps + 1)
    sim_paths[0 :] = spot

    #GBM
    for t in range(n_steps):
        Z = np.random.randn(n_trails)
        sim_paths[t+1, :]= sim_paths[t, :] * np.exp((mu - 0.5 * exp_vol**2) * dT + exp_vol * np.sqrt(dT) * Z)

    rew = np.zeros((n_steps, n_trails))
    
    pos_prev = init_pos * np.ones(n_trails)

    pos_next = policy_BSM(sim_paths[0, :] / strike, maturity * np.ones(n_trails), pos_prev, rate, exp_vol)


    # Hedging loop 
    for timeidx in range(1, n_steps + 1):
        maturity_prev = maturity - sim_times[timeidx - 1]
        maturity_next = np.maximum(0, maturity- sim_times[timeidx])

        # P&L
        pnl_stock = (sim_paths[timeidx, :] - sim_paths[timeidx - 1, :] * pos_prev)
        transaction_cost = np.abs(pos_next - pos_prev) * sim_paths[timeidx, :] * kappa

        # Is this correct way of calculating it? 
        options_change = (- bs_price(sim_paths[timeidx, :], strike, rate, maturity_next, exp_vol)
                          + bs_price(sim_paths[timeidx - 1, :], strike, rate, maturity_prev, exp_vol))
        rew[timeidx - 1, :] = pnl_stock - transaction_cost + options_change

        if timeidx == n_steps:
            rew[timeidx - 1, :] -= pos_next * sim_paths[timeidx, :] * kappa
        else: 
            pos_prev = pos_next
            pos_next = policy(sim_paths[timeidx, :] / strike, maturity_next, pos_prev)

        
    per_costs = np.sum(rew, axisi = 0)
    return per_costs

def compute_cost_2(self, action, policy, n_trails, n_steps, spot, strike, maturity, rate, exp_vol, init_pos, dT, mu, kappa):
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