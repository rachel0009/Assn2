import gymnasium
import aisd_examples
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import os

# Optional: set to "human" for visual, "rgb_array" if you want to render frames
env = gymnasium.make("aisd_examples/RedBall-V0", render_mode="human")

# Initialize and train PPO model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)

# Save the model
model.save("ppo_redball")
del model

# Reload the model
model = PPO.load("ppo_redball")

# Re-create the environment for evaluation (no render to speed things up)
eval_env = gymnasium.make("aisd_examples/RedBall-V0")

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
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, marker='o', linestyle='-')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO Agent Evaluation - Episode Rewards")
plt.grid(True)
plt.tight_layout()
plt.savefig("ppo_agent_evaluation_rewards.png")
plt.show()

