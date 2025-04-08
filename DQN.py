import gymnasium
import aisd_examples
from stable_baselines3 import DQN

env = gymnasium.make("aisd_examples/RedBall-V0")
model = DQN("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn")
del model
model = DQN.load("dqn")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
