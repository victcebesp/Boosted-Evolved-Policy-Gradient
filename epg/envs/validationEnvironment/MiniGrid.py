import gym
import numpy as np
from gym import Env
import gym_minigrid
from gym.spaces import Box

from epg.envs.envPicker.MiniGridEnvironmentPicker import MiniGridEnvironmentPicker
from epg.envs.validationEnvironment.ValidationEnv import ValidationEnv


class MiniGrid(Env, ValidationEnv):

    def get_random_validation_env(self):
        return self.environment_picker.get_validation_environment()

    def __init__(self, seed=None, **_):
        self.environment_picker = MiniGridEnvironmentPicker('MiniGrid-Empty-5x5-v0')
        self.env = self.environment_picker.get_training_environment()
        self.observation_space = Box(low=0, high=5, shape=self.get_flat_shape(self.env))
        self.action_space = self.env.action_space
        self.reset_model = self.meta_reset

    def meta_reset(self, seed): # Take env from the training set
        np.random.seed(seed)

        env = self.environment_picker.get_training_environment()
        env.seed(seed)
        self.env = env

    def get_flat_shape(self, env):
        return (env.observation_space.shape[0] *
                env.observation_space.shape[1] *
                env.observation_space.shape[2], 1)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs.flatten(), reward, done, info

    def reset(self):
        return self.env.reset().flatten()

    def render(self, mode='human'):
        self.env.render(mode)
