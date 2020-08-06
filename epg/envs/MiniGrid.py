import gym
import numpy as np
from gym import Env
import gym_minigrid
from gym.spaces import Box
from gym_minigrid.wrappers import ImgObsWrapper


class MiniGrid(Env):

    def __init__(self, seed=None, **_):
        env = ImgObsWrapper(gym.make('MiniGrid-Empty-8x8-v0'))
        self.env = RandomMiniGrid()
        self.observation_space = Box(low=0, high=5, shape=self.get_flat_shape(env))
        self.action_space = env.action_space
        self.reset_model = self.meta_reset
        self.step = self.env.step
        self.reset = self.env.reset

    def meta_reset(self, seed):
        np.random.seed(seed)

        env = RandomMiniGrid()
        env.seed(seed)

        self.env = env

        # Fix for done flags.
        self.env.reset()
        self.step = env.step
        self.render = env.render
        self.reset = env.reset

    def get_flat_shape(self, env):
        return (env.observation_space.shape[0] *
                env.observation_space.shape[1] *
                env.observation_space.shape[2], 1)


class RandomMiniGrid(Env):

    def __init__(self):
        #self._env = ImgObsWrapper(gym.make('MiniGrid-Empty-Random-5x5-v0'))
        self._env = ImgObsWrapper(gym.make('MiniGrid-Empty-8x8-v0'))

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs.flatten(), reward, done, info

    def reset(self):
        return self._env.reset().flatten()

    def render(self, mode='human'):
        self._env.render(mode)
















