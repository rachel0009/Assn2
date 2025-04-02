import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2

class RedBall(Node):
  """
  A Node to analyse red balls in images and publish the results
  """
  def __init__(self):
    super().__init__('redball')
    self.subscription = self.create_subscription(
      Image,
      'custom_ns/camera1/image_raw',
      self.listener_callback,
      10)
    self.subscription # prevent unused variable warning

    # A converter between ROS and OpenCV images
    self.br = CvBridge()
    self.target_publisher = self.create_publisher(Image, 'target_redball', 10)
    self.twist_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

  def listener_callback(self, msg):
    frame = self.br.imgmsg_to_cv2(msg)

    # convert image to BGR format (red ball becomes blue)
    hsv_conv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    bright_red_lower_bounds = (110, 100, 100)
    bright_red_upper_bounds = (130, 255, 255)
    bright_red_mask = cv2.inRange(hsv_conv_img, bright_red_lower_bounds, bright_red_upper_bounds)

    blurred_mask = cv2.GaussianBlur(bright_red_mask,(9,9),3,3)
# some morphological operations (closing) to remove small blobs
    erode_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate_element = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    eroded_mask = cv2.erode(blurred_mask,erode_element)
    dilated_mask = cv2.dilate(eroded_mask,dilate_element)

    # on the color-masked, blurred and morphed image I apply the cv2.HoughCircles-method to detect circle-shaped objects
    detected_circles = cv2.HoughCircles(dilated_mask, cv2.HOUGH_GRADIENT, 1, 150, param1=100, param2=20, minRadius=2, maxRadius=2000)
    the_circle = None
    if detected_circles is not None:
        for circle in detected_circles[0, :]:
            circled_orig = cv2.circle(frame, (int(circle[0]), int(circle[1])), int(circle[2]), (0,255,0),thickness=3)
            the_circle = (int(circle[0]), int(circle[1]))
        self.target_publisher.publish(self.br.cv2_to_imgmsg(circled_orig))
    else:
        self.get_logger().info('no ball detected')

class RedBallEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):

        rclpy.init(args=None)

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
        
