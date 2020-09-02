import math
import random

import gym
from gym_minigrid.wrappers import ImgObsWrapper

from epg.envs.environmentManager.testEnvironmentManager.TestEnvironmentManager import TestEnvironmentManager
from epg.envs.extendedEnvironment.MiniGrid import MiniGrid


class FourRoomsMinigridEnvironmentManager(TestEnvironmentManager):

    def __init__(self, training_percentage=0.7):
        assert training_percentage < 1
        self.env_id = "MiniGrid-FourRooms-v0"
        self.test_positions_list = self._get_all_positions()

    def get_test_environment(self):
        env = ImgObsWrapper(gym.make(self.env_id))
        env.agent_start_pos = self._get_random_test_position()
        return MiniGrid(env)

    def _get_random_test_position(self):
        return random.choice(self.test_positions_list)

    def _get_all_positions(self):

        all_positions = []

        env = gym.make(self.env_id)
        for idx, grid_object in enumerate(env.grid.grid):
            if grid_object is None:
                row = math.floor(idx / env.width)
                column = idx % env.unwrapped.width
                all_positions.append((row, column))

        return all_positions





