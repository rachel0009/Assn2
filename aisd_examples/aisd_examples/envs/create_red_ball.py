import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from rclpy.node import Node

class RedBallEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):

        rclpy.init(args=args)

        redball = RedBall()

        rclpy.spin(redball)


        self.size = size  # The size of the square grid
        self.window_size = 256  # The size of the PyGame window

        self.states = {}
        self.actions_dict = {}

        self.state = 0

        # Define your observation space as a simple Discrete space
        # of integers, using the length of your dictionary from Step 
        # c above.
        self.observation_space = spaces.Dict({
            "state": spaces.Discrete(1)
        })

        self.action_space = spaces.Discrete(1)

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"state":  self.state}

    def _get_info(self):
        info = {
            "state": self.state, 
        }
        return info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action=1):
        action = int(action)

        reward = 0
        
        terminated = False
        reward = 100 if terminated else reward

        observation = self._get_obs()
        info = self._get_info()

        self.render()

        return observation, reward, terminated, False, info

    def render(self):
        return

    def close(self):
        redball.destroy_node()
        rclpy.shutdown()
        
