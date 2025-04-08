import gymnasium
import aisd_examples
import matplotlib.pyplot as plt
import os

env_name = "aisd_examples/RedBall-V0"
env = gymnasium.make(env_name, render_mode="human")

observation, info = env.reset()

episodes = 50
episode_returns = []

def choose_action(position):
    if position < 320:
        return position + 10  # move right
    elif position > 320:
        return position - 10  # move left
    else:
        return 320  # stay
    
for episode in range(episodes):
    observation, info = env.reset()
    total_reward = 0

    done = False
    step = 0
    while not done:
        os.system('clear')
        print(f"Episode {episode+1}: Return = {total_reward:.3f}")
        position = observation['position']
        action = choose_action(position)

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

    print(f"Episode {episode+1}: Return = {total_reward:.3f}")
    episode_returns.append(total_reward)

env.close()

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(episodes), episode_returns, marker='o', linestyle='-')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Episode Returns")

plt.tight_layout()
plt.show()