from collections import deque
import numpy as np

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



# def train_hybrid(episodes, env, dqn_agent, td3_agent, batch_size, actions_list, score_window, stop_avg_reward):
#     """
#     Hybrid Training: DQN selects the range, TD3 selects the specific value.
#     """
#     all_episode_rewards = []
#     score_window = deque(maxlen=200) 

#     for episode in range(episodes):
#         state = env.reset() #
#         episode_reward = 0
#         done = False
        
#         while not done:
#             # 1. DQN selects the range (bin index)
#             # Using the epsilon-greedy selection logic from your dqn_agent.py
#             bin_idx = dqn_agent.select(state, train=True)
            
#             # Define bounds based on your actions_list
#             lower_bound = actions_list[bin_idx]
#             upper_bound = actions_list[bin_idx + 1] if bin_idx + 1 < len(actions_list) else 1.0
            
#             # 2. TD3 selects a value in [0, 1]
#             # Using the actor network from your td3_agent.py
#             raw_td3_action = td3_agent.select(state)
            
#             # 3. Combine: Rescale TD3 output into the DQN bin
#             # Final Action = Lower + (Relative_Pos * Width)
#             action = lower_bound + (raw_td3_action * (upper_bound - lower_bound))
            
#             # 4. Environment Step
#             reward, next_state, done = env.step(action) #
            
#             # 5. Store and Train
#             # Note: DQN stores the bin index, TD3 stores the raw/rescaled action
#             dqn_agent.buffer.add(state, bin_idx, reward, next_state, done) #
#             td3_agent.buffer.add(state, raw_td3_action, reward, next_state, done) #
            
#             dqn_agent.train(batch_size) #
#             td3_agent.train(batch_size) #
            
#             state = next_state
#             episode_reward += reward

#         # Decay noise and log results
        
#         all_episode_rewards.append(episode_reward)
#         score_window.append(episode_reward)
#         avg_reward = np.mean(score_window)

#         if episode % 100 == 0:
#             print(f"Hybrid Episode {episode}, Reward {episode_reward:.4f}, Avg {avg_reward:.4f}")

#         if avg_reward > stop_avg_reward and len(score_window) == score_window.maxlen:
#             print("Stopping: Average reward threshold reached")
#             break

#     return all_episode_rewards

# def train_hybrid1(episodes, env, hybrid_agent, batch_size, actions_list, score_window, stop_avg_reward): 


#     all_episode_rewards = []

#     # I din main-fil eller träningsloop:
#     # --- STEG 1: TRÄNA BARA DQN ---
#     print("Fas 1: Tränar DQN för grova intervall...")
#     for episode in range(500): # Exempelvis 500 episoder
#         state = env.reset()
#         episode_reward = 0
#         hybrid_agent.td3.reset_noise()
#         # ... vanliga träningssteg ...
#         done = False
#         while not done:
#             action_idx = hybrid_agent.dqn.select(state)
#             base_hedge = actions_list[action_idx]


#             state[2] = base_hedge # så ska det inte vara, vi behöver 4, så att vi vet hur mycket td3 vill ändra. 
#             reward, next_state, done = env.step(base_hedge)
#             hybrid_agent.dqn.buffer.add(state, action_idx, reward, next_state, done)
#             #hybrid_agent.dqn.train(batch_size)


#             raw_delta = hybrid_agent.td3.select_ou(state)
        
#             # 3. Få finjustering från TD3
#             # TD3 bör tränas att ge ett litet delta, t.ex. mellan -0.05 och 0.05
#             # 4. TRANSFORMATION (Viktigt!):s
#             # Vi centrerar 0.5 till att bli 0. 
#             # Ett värde på 0.6 blir +0.02, ett värde på 0.4 blir -0.02.
#             fine_tune = (raw_delta - 0.5) * 0.2  # Ger max +/- 0.1 i justering


#             # 4. Kombinera och klicka (clip) mellan 0 och 1
#             final_action = np.clip(base_hedge + fine_tune, 0.0, 1.0)

#             hybrid_agent.dqn.buffer.add(state, action_idx, reward, next_state, done)
#             hybrid_agent.dqn.train(batch_size)


            
#             # Under fas 1 kan vi låta TD3 ge 0 i tillägg
#             reward, next_state, done = env.step(base_hedge)
#             state[2] = base_hedge
            
#             hybrid_agent.dqn.buffer.add(state, action_idx, reward, next_state, done)
#             hybrid_agent.dqn.train(batch_size)

#             hybrid_agent.select(state)


#             state = next_state
#             episode_reward += reward
            
#             all_episode_rewards.append(episode_reward)

#         # Logging & stopping 
#         score_window.append(episode_reward)
#         avg_reward = np.mean(score_window)
        
#         if episode % 100 == 0:
#             print(f"DQN + TD3 Episode {episode}, Reward {episode_reward:.4f}, Avg {avg_reward:.4f}")


#         if avg_reward > stop_avg_reward and len(score_window) == score_window.maxlen:
#             print("Stopping: Average reward threshold reached")
#             break

#     # --- STEG 2: TRÄNA TD3 MED LÅST DQN ---
#     print("Fas 2: Låser DQN och tränar TD3 för finjustering...")
#     for episode in range(500, 1000):
#         #state = env.reset()
#         hybrid_agent.td3.reset_noise()
        
#         done = False
#         while not done:
#             # DQN väljer bas (ingen träning för DQN här!)
#             action_idx = hybrid_agent.dqn.select(state) 
#             base_hedge = actions_list[action_idx]
            
#             # TD3 väljer delta baserat på vad DQN valde
#             #augmented_state = np.append(state[2], base_hedge) [1.0,0,256, 0,2]
#             state[2] = base_hedge
#             delta = hybrid_agent.select(state)
            
#             final_action = np.clip(delta, 0, 1)
#             reward, next_state, done = env.step(final_action)
            
#             # Spara i TD3:s buffer och träna bara TD3
#             hybrid_agent.td3.buffer.add(state, delta, reward, next_state, done)
#             hybrid_agent.td3.train(batch_size)
            
#             state = next_state
#             episode_reward += reward
        
#         all_episode_rewards.append(episode_reward)

#         # Logging & stopping 
#         score_window.append(episode_reward)
#         avg_reward = np.mean(score_window)

#         if episode % 100 == 0:
#             print(f"DQN + TD3 Episode {episode}, Reward {episode_reward:.4f}, Avg {avg_reward:.4f}")


#         if avg_reward > stop_avg_reward and len(score_window) == score_window.maxlen:
#             print("Stopping: Average reward threshold reached")
#             break

#     return all_episode_rewards