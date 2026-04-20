import numpy as np
from collections import deque

def train_DQN(episodes, env, agent, batch_size, actions_list, score_window_length, stop_avg_reward):

    all_episode_rewards = []
    score_window = deque(maxlen=score_window_length) 

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Run until time to maturity is reached
        while not done:
            # Send noise when we choose action
            action_idx = agent.select(state)
            action = actions_list[action_idx]
            reward, next_state, done = env.step(action)
            
            agent.buffer.add(state, action_idx, reward, next_state, done)
            agent.train(batch_size)
            
            state = next_state
            episode_reward += reward

        # Decay (minska bruset gradvis över episoderna)
        
        all_episode_rewards.append(episode_reward)

        # Logging & stopping 
        score_window.append(episode_reward)
        avg_reward = np.mean(score_window)

        if episode % 100 == 0:
            print(f"DQN Episode {episode}, Reward {episode_reward:.4f}, Avg {avg_reward:.4f}")


        if avg_reward > stop_avg_reward and len(score_window) == score_window.maxlen:
            print("Stopping: Average reward threshold reached")
            break

    return all_episode_rewards