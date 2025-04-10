import os
import gymnasium
import aisd_examples
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np

from config import ENV_NAME, EPISODES

def train_ppo():
    #env = Monitor(gymnasium.make(ENV_NAME))

    #eval_callback = EvalCallback(
    #    env,
    #    best_model_save_path="./logs/",
    #    log_path="./logs/",
    #    eval_freq=1000,
    #    deterministic=True,
    #    render=False
    #)

    #model = PPO("MlpPolicy", env, verbose=1)
    #model.learn(total_timesteps=10000, callback=eval_callback)
    #model.save("ppo_redball")

    # Close environments
    #env.close()

    data = np.load("./logs/evaluations.npz")
    timesteps = data["timesteps"]
    episode_rewards = data["results"].mean(axis=1)

    plt.figure(figsize=(12, 5))
    plt.plot(timesteps, episode_rewards, marker='o', linestyle='-')
    plt.title("Training Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Returns - PPO")
    plt.tight_layout()
    plt.savefig("ppo.png")
    plt.show()

if __name__ == "__main__":
    train_ppo()
