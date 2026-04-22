from collections import deque
import numpy as np
from train.train_DQN import train_DQN

def train_hybrid(episodes, env, agent, batch_size, score_window_length, stop_avg_reward):
    """
    Training loop for the HybridAgent (DQN + TD3).

    At each step, the agent selects a hedge ratio using its hierarchical
    policy: DQN selects a coarse bin, TD3 fine-tunes within that bin.
    Both sub-agents store their respective transitions independently and
    are trained from their own replay buffers after each step.
    """
    all_episode_rewards = []
    score_window = deque(maxlen=score_window_length)

    for episode in range(episodes):
        state = env.reset()
        agent.reset_noise()
        episode_reward = 0
        done = False

        while not done:
            # Agent selects action using DQN (coarse) + TD3 (fine)
            # and stores the transition in each sub-agent's replay buffer
            action, bin_idx, raw_td3 = agent.select(state)
            reward, next_state, done = env.step(action)

            # DQN learns to select the right bin — stores bin index as action
            agent.dqn.buffer.add(state, bin_idx, reward, next_state, done)

            # TD3 learns to fine-tune within the bin — stores its raw output
            agent.td3.buffer.add(state, raw_td3, reward, next_state, done)

            # Experience Buffer
            agent.experience_buffer.add(state, bin_idx, raw_td3, reward, next_state, done)

            # Train both sub-agents
            agent.train(batch_size)

            state = next_state
            episode_reward += reward

        all_episode_rewards.append(episode_reward)
        score_window.append(episode_reward)
        avg_reward = np.mean(score_window)

        if episode % 100 == 0:
            print(f"{agent} Episode {episode}, Reward {episode_reward:.4f}, Avg {avg_reward:.4f}")

        if avg_reward > stop_avg_reward and len(score_window) == score_window.maxlen:
            print("Stopping: Average reward threshold reached")
            break

    return all_episode_rewards

def train_hybrid_sequential(episodes_dqn, episodes_td3, env, agent, batch_size, score_window_length, stop_avg_reward):
    """
    Two-phase sequential training for the HybridAgent.
    Phase 1: Train DQN alone to learn coarse bin selection.
    Phase 2: Freeze DQN. Train TD3 to fine-tune within the bin DQN selects.           
    """

    # Phase 1: Train DQN ───
    print("Phase 1: Training DQN for coarse bin selection...")

    dqn_rewards = train_DQN(
        int(episodes_dqn), env, agent.dqn,
        batch_size, agent.actions_list,
        score_window_length, stop_avg_reward
    )

    print("Phase 1 complete. Freezing DQN.\n")

    # Phase 2: Train TD3 with DQN frozen
    print("Phase 2: Training TD3 for fine-grained adjustment...")

    td3_rewards = []
    score_window = deque(maxlen=score_window_length)

    for episode in range(int(episodes_td3)):
        state = env.reset()
        agent.reset_noise()
        episode_reward = 0
        done = False

        while not done:
            # DQN selects bin 
            bin_idx = agent.dqn.select(state, train=False)
            lower_bound = agent.actions_list[bin_idx]
            upper_bound = (
                agent.actions_list[bin_idx + 1]
                if bin_idx + 1 < len(agent.actions_list)
                else 1.0
            )

            # TD3 fine-tunes within the bin
            raw_td3 = agent.td3.select(state)
            action = np.clip(
                lower_bound + raw_td3 * (upper_bound - lower_bound),
                0.0, 1.0
            )

            reward, next_state, done = env.step(action)

            # Only TD3's buffer is updated, DQN is frozen
            agent.td3.buffer.add(state, raw_td3, reward, next_state, done)
            agent.td3.train(batch_size)

            state = next_state
            episode_reward += reward

        td3_rewards.append(episode_reward)
        score_window.append(episode_reward)
        avg_reward = np.mean(score_window)

        if episode % 100 == 0:
            print(f"TD3 fine-tune Episode {episode}, "
                  f"Reward {episode_reward:.4f}, Avg {avg_reward:.4f}")

        if avg_reward > stop_avg_reward and len(score_window) == score_window.maxlen:
            print("Stopping: Average reward threshold reached")
            break

    print("Phase 2 complete.")
    return dqn_rewards, td3_rewards