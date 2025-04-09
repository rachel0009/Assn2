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
    # Position: Left 1, Center 0, Right 2, 
    # Not Detected 3
    # Action: Left 1, Right 2, No Movement 0
    if position == 3:
        return 0  
    else:
        return position  
    
for episode in range(episodes):
    observation, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        os.system('clear')
        print(f"Episode {episode+1}: Return = {total_reward:.3f}")
        position = observation
        action = choose_action(position)

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Episode {episode+1}: Return = {total_reward:.3f}")
    episode_returns.append(total_reward)

env.close()

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(episodes), episode_returns, marker='o', linestyle='-')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Episode Returns - NonRL")

plt.tight_layout()
plt.show()