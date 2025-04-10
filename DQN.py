import os
import gymnasium
import aisd_examples
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

from config import ENV_NAME, EPISODES

def train_ppo(render = False):
    env = gymnasium.make(ENV_NAME)

    # Initialize and train PPO model
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)

    # Save the model
    model.save("dqn_redball")
    del model

    # Reload the model
    model = DQN.load("dqn_redball")

    # Re-create the environment for evaluation (no render to speed things up)
    eval_env = gymnasium.make(ENV_NAME)

    # Evaluate the policy and get individual episode rewards
    episode_rewards = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=50,
        deterministic=True,
        return_episode_rewards=True
    )

    # Close environments
    env.close()
    eval_env.close()

    # Plot the evaluation episode rewards
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, marker='o', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Returns - DQN")

    plt.tight_layout()
    plt.savefig("dqn.png")
    plt.show()