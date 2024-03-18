from IPython.display import clear_output
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchrl.envs import *
from torchrl.envs.libs.gym import *
from joblib import Parallel, delayed
from Constants.constants import *

import gym
import numpy as np
import cv2


class CartPoleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # modify obs
        return obs

class ImageCartPole(gym.Env):
    def __init__(self, width=W, height=H):
        self.env = gym.make('CartPole-v1')
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(height, width, 1), dtype=np.float32)
        self.action_space = self.env.action_space

    def reset(self):
        self.env.reset()
        return self._get_observation()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        # Render the environment
        screen = self.env.render(mode='rgb_array')

        # Convert to grayscale
        screen_gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

        # Resize the screen
        screen_resized = cv2.resize(screen_gray, (self.width, self.height))

        # Normalize pixel values to range [0, 1]
        screen_normalized = screen_resized / 255.0

        # Add channel dimension
        screen_normalized = np.expand_dims(screen_normalized, axis=-1)

        return screen_normalized

    def close(self):
        self.env.close()

# Example usage
# env = ImageCartPole()
# observation = env.reset()

# for _ in range(1000):
#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)
#     if done:
#         observation = env.reset()
# env.close()


