import gym
import numpy as np
from gym import Env
from gym.spaces import Box
from gym_minigrid.wrappers import ImgObsWrapper

from epg.envs.validationEnvironment.ValidationEnv import ValidationEnv


class MiniGrid(Env, ValidationEnv):

    def __init__(self, env=None, seed=None, **_):
        print(type(env))
        self.env = ImgObsWrapper(gym.make('MiniGrid-DoorKey-5x5-v0')) if env is None else env
        self.observation_space = Box(low=0, high=5, shape=self.get_flat_shape(self.env))
        self.action_space = self.env.action_space
        self.reset_model = self.meta_reset

    def get_optimal_episode_length(self):
        start_position = self.env.unwrapped.agent_start_pos
        end_position = (self.env.unwrapped.width - 2, self.env.unwrapped.width - 2)
        optimal_ep_length = abs(start_position[0] - end_position[0]) + abs(start_position[1] - end_position[1])
        return optimal_ep_length

    def meta_reset(self, seed):
        np.random.seed(seed)
        self.env.reset()

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
