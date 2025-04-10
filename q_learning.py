import os
import gymnasium
import aisd_examples
import numpy as np
import time
import matplotlib.pyplot as plt
from config import ENV_NAME, EPISODES

def train_q_learning():
    env = gymnasium.make(ENV_NAME)
    
    # Number of states and actions
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Initialize Q-table
    qtable = np.random.rand(num_states, num_actions)

    # Hyperparameters
    episodes = EPISODES
    gamma = 0.1
    epsilon = 0.08
    decay = 0.1
    alpha = 1

    # For tracking performance
    episode_rewards = []

    for i in range(episodes):
        obs, _ = env.reset()
        state = int(np.argmax(obs))
        total_reward = 0
        done = False

        while not done:
            os.system('clear')
            print("Running Q-Learning...")
            print(f"Episode {i+1}: Return = {total_reward:.3f}")
            time.sleep(0.05)

            # ε-greedy action selection
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state])

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = int(np.argmax(next_obs))

            # Q-learning update rule
            qtable[state][action] += alpha * (reward + gamma * np.max(qtable[next_state]) - qtable[state][action])

            state = next_state
            total_reward += reward
            done = terminated or truncated

        # Decay exploration rate
        epsilon -= decay * epsilon

        episode_rewards.append(total_reward)

        print(f"Accumulated Reward: {total_reward}")
        time.sleep(1)

    env.close()

    # Save the graph
    plt.figure(figsize=(12, 5))
    plt.plot(range(episodes), episode_rewards, marker='o', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Returns - QLearning")
    plt.tight_layout()
    plt.savefig("qlearning.png")

if __name__ == "__main__":
    train_q_learning()
