# Hedging using Deep reinforcement learning
Master's thesis in Industial Enginering and Management, Umeå University (2026).
Authors: Felicia Aronsson & Jelena Nääs

## Overview
This repository contains the implementation for our master's thesis, which investigates whether deep reinforcement learning agents can learn 
hedging startegies that outperforms the classical Black-Scholes delta-hedging, in markets with transaction costs.

## Environment 
The hedging environment (`env/hedging_env.py`) simulates the problem of dynamically hedging a European call option. At each time step the agent observes a state of three variables: Moneyness (spot price / strike price (S/K)), Time to maturity, Current position (the hedge ratio currently held). The agent selects a new hedge ratio and receives a reward based on the accounting P&L, adjusted for transaction costs and a quadratic risk penalty. The underlying asset price follows Geometric Brownian Motion.

## Agents 
DQN: Discretises the hedge ratio into 11 evenly spaced values {0.0, 0.1, …, 1.0} and applies Q-learning with a target network and epsilon-greedy exploration. 

DDPG: Actor–critic architecture for continuous actions. Uses an Ornstein–Uhlenbeck noise process for exploration during training. 

TD3: Extends DDPG with three improvements: clipped double Q-learning (two critic networks, take the minimum), target policy smoothing, and delayed policy updates. More stable than DDPG in practice. 

Hybrid (DQN + TD3): Combines DQN and TD3. DQN is trained to select a coarse bin in the action space. TD3 is trained to fine-tune within the DQN-selected bin.

## Run main
This will: 
1. Train all four agents for the configured number of episodes 
2. Evaluate each agent on 1,000 simulated price paths 
3. Print a hedging cost comparison table and produce a HTML report
4. Display learning curves and cost distribution histograms