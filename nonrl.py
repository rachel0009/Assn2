import os
import gymnasium
import aisd_examples
import matplotlib.pyplot as plt
import numpy as np
from config import ENV_NAME, EPISODES

def choose_action(position):
    # Position: Left 1, Center 0, Right 2, 
    # Not Detected 3
    # Action: Left 1, Right 2, No Movement 0
    if position == 3:
        return 0  
    else:
        return position  
    
def run_non_rl():
    env = gymnasium.make(ENV_NAME)
    episodes = EPISODES
    episode_returns = []

    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            os.system('clear')
            print("Running Non-RL...")
            print(f"Episode {episode+1}: Return = {total_reward:.3f}")
            state = int(np.argmax(obs))
            action = choose_action(state)
            print(f"Ball Position is at {state}, action made is {action}")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        episode_returns.append(total_reward)

    env.close()

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.plot(range(episodes), episode_returns, marker='o', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Returns - NonRL")

    plt.tight_layout()
    plt.savefig("nonrl.png") 
    plt.show()

if __name__ == "__main__":
    run_non_rl()