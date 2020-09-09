import math

import gym
import numpy as np
from gym import Env
from gym.spaces import Box
from gym_minigrid.minigrid import Goal
from gym_minigrid.wrappers import ImgObsWrapper

from epg.envs.extendedEnvironment.ExtendedEnvironment import ExtendedEnvironment

# MiniGrid extended environment
class MiniGrid(Env, ExtendedEnvironment):

    def __init__(self, env=None, seed=None, **_):
        self.env = ImgObsWrapper(gym.make('MiniGrid-Empty-5x5-v0')) if env is None else env
        self.observation_space = Box(low=0, high=5, shape=self.get_flat_shape(self.env))
        self.action_space = self.env.action_space
        self.reset_model = self.meta_reset
        self.goal_position = self.get_goal_position()

    # Returns the optimal episode's length using the Manhattan distance
    def get_optimal_episode_length(self):
        start_position = self.env.agent_start_pos
        end_position = self.get_goal_position()
        optimal_ep_length = abs(start_position[0] - end_position[0]) + abs(start_position[1] - end_position[1])
        return optimal_ep_length

    # Overwrite meta_reset method to reset the environment and reseed numpy random generation
    def meta_reset(self, seed):
        np.random.seed(seed)
        self.env.reset()

    # Return the flat shape of the environment's states
    def get_flat_shape(self, env):
        return (env.observation_space.shape[0] *
                env.observation_space.shape[1] *
                env.observation_space.shape[2], 1)

    # Overwrite step method to return a flat version of the observations
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs.flatten(), reward, done, info

    # Overwrite reset method to clean the default goal position
    def reset(self):
        obs = self.env.reset().flatten()
        self.clean_default_goal()
        self.env.unwrapped.put_obj(Goal(), self.goal_position[0], self.goal_position[1])
        return obs

    # Overwrite render method to call the render method of the saved env in the object variable env
    def render(self, mode='human'):
        self.env.render(mode)

    # Returns the position of the goal by inspecting the current grid of the environment
    def get_goal_position(self):
        for idx, grid_object in enumerate(self.env.unwrapped.grid.grid):
            if isinstance(grid_object, Goal):
                return math.floor(idx / self.env.unwrapped.width), idx % self.env.unwrapped.width

    # Cleans the goal initial position by inspecting the current grid of the environment
    def clean_default_goal(self):
        for i, grid_object in enumerate(self.env.unwrapped.grid.grid):
            if isinstance(grid_object, Goal):
                self.env.unwrapped.grid.grid[i] = None
