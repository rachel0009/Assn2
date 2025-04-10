import os
import gymnasium
import aisd_examples
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np

from config import ENV_NAME, EPISODES

def train_dqn():
    #env = Monitor(gymnasium.make(ENV_NAME))

    #eval_callback = EvalCallback(
    #    env,
    #    best_model_save_path="./dqn_logs/",
    #    log_path="./dqn_logs/",
    #    eval_freq=EPISODES,
    #    deterministic=True,
    #    render=False
    #)

    #model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_tensorboard/")
    #model.learn(total_timesteps=10000, callback=eval_callback)
    #model.save("dqn_redball")

    # Close environments
    #env.close()

    data = np.load("./dqn_logs/evaluations.npz")
    timesteps = data["timesteps"]
    episode_rewards = data["results"].mean(axis=1)

    plt.figure(figsize=(12, 5))
    plt.plot(timesteps, episode_rewards, marker='o', linestyle='-')
    plt.title("Training Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Returns - DQN")
    plt.tight_layout()
    plt.savefig("dqn.png")
    plt.show()

if __name__ == "__main__":
    train_dqn()
