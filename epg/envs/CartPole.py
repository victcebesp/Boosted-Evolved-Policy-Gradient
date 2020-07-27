import gym
import numpy as np
from gym import Env, utils


class CartPole(Env):

    def __init__(self, seed=None, **_):
        env = gym.make('CartPole-v0')
        self.env = RandomCartPole()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reset_model = self.meta_reset
        self.step = env.step
        self.reset = env.reset

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


class RandomCartPole(Env):

    def __init__(self):
        self._env = gym.make('CartPole-v0')
        self._env._max_episode_steps = 500

    def step(self, action):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()

    def render(self, mode='human'):
        self._env.render(mode)
















