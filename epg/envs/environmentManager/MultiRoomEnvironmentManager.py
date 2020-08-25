import gym
from gym_minigrid.wrappers import ImgObsWrapper

from epg.envs.environmentManager.TestEnvironmentManager import TestEnvironmentManager
from epg.envs.validationEnvironment.MiniGrid import MiniGrid


class MultiRoomEnvironmentManager(TestEnvironmentManager):

    def get_test_environment(self):

        env = ImgObsWrapper(gym.make('MiniGrid-MultiRoom-N2-S4-v0'))
        return MiniGrid(env)
