import argparse
import gymnasium
import aisd_examples

env_name = "aisd_examples/RedBall-V0"
env = gymnasium.make(env_name, render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
