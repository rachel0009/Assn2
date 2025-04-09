import gymnasium
import aisd_examples
from stable_baselines3 import PPO
import os
import matplotlib.pyplot as plt

env = gymnasium.make("aisd_examples/RedBall-V0")

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=50, log_interval=4)

model.save("ppo")
del model 
model = PPO.load("ppo")

episodes = 50
episode_rewards = []

for i in range(episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Episode {i+1}: Return = {total_reward:.3f}")
    episode_rewards.append(total_reward)

env.close()

# Plot the total rewards
plt.figure(figsize=(8, 5))
plt.plot(range(1, episodes + 1), episode_rewards, marker='o', linestyle='-')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Episode Returns - PPO")
plt.grid(True)
plt.savefig("ppo_agent_returns.png")  
plt.show()