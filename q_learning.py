import gymnasium
import aisd_examples
import numpy as np
import time
import os
import matplotlib.pyplot as plt

env = gymnasium.make("aisd_examples/RedBall-V0")

observation, info = env.reset()

# Number of states: 641 possible positions for the red ball (from 0 to 640)
num_states = env.observation_space.n
num_actions = env.action_space.n  # Number of possible actions

# QTable : contains the Q-Values for every (state,action) pair
qtable = np.random.rand(num_states, num_actions).tolist()

# hyperparameters
episodes = 50
gamma = 0.1
epsilon = 0.08
decay = 0.1
alpha = 1

# Tracking performance
episode_rewards = []
steps_per_episode = []

for i in range(episodes):
    state_dict, info = env.reset()
    state = state_dict['position']  # State is directly the red ball's position
    
    total_reward = 0
    done = False

    while not done:
        os.system('clear')
        print(f"Episode {i+1}: Return = {total_reward:.3f}")
        env.render()
        time.sleep(0.05)

         # act randomly sometimes to allow exploration
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        # if not select max action in Qtable (act greedy)
        else:
            action = np.argmax(qtable[state])


        # Take action
        next_state_dict, reward, done, truncated, info = env.step(action)
        next_state = next_state_dict['position'] # Next state is the red ball's new position
        
        # Update total reward
        total_reward += reward

        # Q-learning update using Sutton's rule
        qtable[state][action] += alpha * (reward + gamma * max(qtable[next_state]) - qtable[state][action])
        
        # Update state
        state = next_state

        # Handle episode termination
        done = done or truncated
    
    # Decay epsilon to reduce exploration over time
    epsilon -= decay * epsilon

    # Store episode results
    episode_rewards.append(total_reward)

    print(f"Accumulated Reward: {total_reward}")
    time.sleep(1)

env.close()

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(episodes), episode_rewards, marker='o', linestyle='-')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Episode Returns - QLearning")

plt.tight_layout()
plt.show()
