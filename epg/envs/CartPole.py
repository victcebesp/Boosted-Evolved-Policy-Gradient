import gym
import numpy as np
from gym import Env, utils
from gym.utils import EzPickle


class CartPole(Env, EzPickle):

    def __init__(self, seed=None, **_):
        utils.EzPickle.__init__(self)
        env = gym.make('CartPole-v0')
        self.env = RandomCartPole()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reset_model = self.meta_reset
        self.step = env.step

    def meta_reset(self, seed):
        np.random.seed(seed)

        env = RandomCartPole()
        env.seed(seed)

        self.env = env

        # Fix for done flags.
        self.env.reset()
        self.step = env.step
        self.render = env.render
        self.reset = env.reset


class RandomCartPole(Env, EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        self._env = gym.make('CartPole-v0')

    def step(self, action):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()

    def render(self, mode='human'):
        self._env.render(mode)
















