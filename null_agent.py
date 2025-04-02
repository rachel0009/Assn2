import argparse
import gymnasium
import aisd_examples

env_name = "aisd_examples/RedBall-V0"
env = gymnasium.make(env_name, render_mode="human")

observation, info = env.reset()

# do a random action 1000 times
while True:
    action = 1
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print(f"Terminated at: {info}")
        observation, info = env.reset()
        print(f"New info: {info}")

env.close()
