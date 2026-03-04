import numpy as np

def train_RL(episodes, env, agent, batch_size, min_noise, noise_scale, noise_decay, score_window, stop_avg_reward):
    """
    Training the enviroment
    
    :param episodes: Description
    :param env: Description
    :param agent: Description
    :param batch_size: Description
    :param min_noise: Description
    :param noise_decay: Description
    :param score_window: Description
    :param stop_avg_reward: Description
    """

    all_episode_rewards = []

    # Training
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Run until time to maturity is reached
        while not done:
            # Send noice when we choose action
            action = agent.select(state, noise_scale)
            reward, next_state, done = env.step(action)
            
            agent.buffer.add(state, action, reward, next_state, done)
            agent.train(batch_size)
            
            state = next_state
            episode_reward += reward

        # Decay (minska bruset gradvis över episoderna)
        noise_scale = max(min_noise, noise_scale * noise_decay)
        all_episode_rewards.append(episode_reward)

        # Logging & stopping 
        score_window.append(episode_reward)
        avg_reward = np.mean(score_window)

        if episode % 100 == 0:
            print(f"DDPG Episode {episode}, Reward {episode_reward:.4f}, Avg {avg_reward:.4f}, Noise {noise_scale:.3f}")


        if avg_reward > stop_avg_reward and len(score_window) == score_window.maxlen:
            print("Stopping: Average reward threshold reached")
            break

    return all_episode_rewards