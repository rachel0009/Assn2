import os
import gymnasium
import aisd_examples
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

from config import ENV_NAME, EPISODES

def train_dqn():
    env = gymnasium.make(ENV_NAME)

    # Initialize and train PPO model
    model = DQN("MlpPolicy", env, verbose=1)
    #model.learn(total_timesteps=10000, log_interval=4)

    # Save the model
    #model.save("dqn_redball")
    #del model

    # Reload the model
    model = DQN.load("dqn_redball")

    # Evaluation loop
    n_episodes = EPISODES
    episode_rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
        episode_rewards.append(total_reward)


    # Close environments
    env.close()

    # Plot the evaluation episode rewards
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, marker='o', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Returns - DQN")

    plt.tight_layout()
    plt.savefig("dqn.png")
    plt.show()

if __name__ == "__main__":
    train_dqn()
