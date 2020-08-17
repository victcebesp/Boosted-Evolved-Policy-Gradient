import random
from math import ceil

import gym
from gym_minigrid.wrappers import ImgObsWrapper

from epg.envs.envPicker.EnvironmentPicker import EnvironmentPicker
from epg.envs.validationEnvironment.MiniGrid import MiniGrid


class MiniGridEnvironmentPicker(EnvironmentPicker):



    def __init__(self, env_id, training_percentage=0.7, validation_percentage=0.15):
        super().__init__(env_id)
        assert training_percentage < 1
        self.training_positions_list, \
            self.validation_positions_list, \
                self.test_positions_list = self._split_training_validation_positions(training_percentage,
                                                                                     validation_percentage)

    def get_training_environment(self):
        env = ImgObsWrapper(gym.make(self.env_id))
        env.agent_start_pos = self._get_random_training_position()
        return MiniGrid(env)

    def get_validation_environment(self):
        env = ImgObsWrapper(gym.make(self.env_id))
        env.agent_start_pos = self._get_random_validation_position()
        return MiniGrid(env)

    def get_test_environment(self):
        env = ImgObsWrapper(gym.make(self.env_id))
        env.agent_start_pos = self._get_random_test_position()
        return MiniGrid(env)

    def _get_random_training_position(self):
        return random.choice(self.training_positions_list)

    def _get_random_validation_position(self):
        return random.choice(self.validation_positions_list)

    def _get_random_test_position(self):
        return random.choice(self.test_positions_list)

    def _split_training_validation_positions(self, training_percentage, validation_percentage):
        grid_width = gym.make(self.env_id).width
        all_positions = []
        for row in range(1, grid_width - 1):
            for column in range(1, grid_width - 1):
                all_positions.append((row, column))

        all_positions.remove((grid_width - 2, grid_width - 2))
        print(len(all_positions))

        training_population_length = ceil(len(all_positions) * training_percentage)
        validation_population_length = int(len(all_positions) * validation_percentage)

        random.shuffle(all_positions)

        training_positions = all_positions[:training_population_length]
        validation_positions = all_positions[training_population_length:training_population_length + validation_population_length]
        test_positions = all_positions[training_population_length + validation_population_length:]

        return training_positions, validation_positions, test_positions
