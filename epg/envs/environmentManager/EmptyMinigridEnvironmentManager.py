import math
import random
from math import ceil

import gym
from gym_minigrid.minigrid import Goal
from gym_minigrid.wrappers import ImgObsWrapper

from epg.envs.environmentManager.TrainingEnvironmentManager import TrainingEnvironmentManager
from epg.envs.environmentManager.ValidationEnvironmentManager import ValidationEnvironmentManager
from epg.envs.extendedEnvironment.MiniGrid import MiniGrid


class EmptyMinigridEnvironmentManager(TrainingEnvironmentManager, ValidationEnvironmentManager):

    def __init__(self, training_percentage=0.7):
        assert training_percentage < 1
        self.env_id = "MiniGrid-Empty-6x6-v0"
        self.training_positions_list, \
            self.validation_positions_list = self._split_training_validation_positions(training_percentage)

    def get_training_environment(self):
        return self._build_environment(*self._get_random_training_position())

    def get_validation_environment(self):
        return self._build_environment(*self._get_random_validation_position())

    def _build_environment(self, goal_position, agent_position):
        env = ImgObsWrapper(gym.make(self.env_id))
        self.clean_default_goal(env)
        env.agent_start_pos = agent_position
        env.unwrapped.put_obj(Goal(), goal_position[0], goal_position[1])
        return MiniGrid(env)

    def _get_random_training_position(self):
        return random.choice(self.training_positions_list)

    def _get_random_validation_position(self):
        return random.choice(self.validation_positions_list)

    def _split_training_validation_positions(self, training_percentage):
        all_positions = self._get_all_positions()

        training_population_length = ceil(len(all_positions) * training_percentage)

        random.shuffle(all_positions)

        training_positions = all_positions[:training_population_length]
        validation_positions = all_positions[
                               training_population_length:]

        return training_positions, validation_positions

    def _get_all_positions(self):

        env = gym.make(self.env_id)

        for i, grid_object in enumerate(env.grid.grid):
            if isinstance(grid_object, Goal):
                env.grid.grid[i] = None

        all_positions = []

        for i in range(len(env.grid.grid)):
            if env.grid.grid[i] is None:
                grid_copy = list(env.grid.grid)
                grid_copy[i] = Goal()
                for j, grid_object in enumerate(grid_copy):
                    if grid_object is None:
                        goal_position = self.transform_to_2D_coordinates(i)
                        agent_position = self.transform_to_2D_coordinates(j)
                        all_positions.append((goal_position, agent_position))

        return all_positions

    def transform_to_2D_coordinates(self, index):

        env = gym.make(self.env_id)

        row = math.floor(index / env.width)
        column = index % env.width
        return (row, column)


    def clean_default_goal(self, env):
        for i, grid_object in enumerate(env.unwrapped.grid.grid):
            if isinstance(grid_object, Goal):
                env.unwrapped.grid.grid[i] = None