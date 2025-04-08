import gymnasium
import aisd_examples
from stable_baselines3 import PPO

env = gymnasium.make("aisd_examples/RedBall-V0")
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("ppo")
del model 
model = PPO.load("ppo")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
