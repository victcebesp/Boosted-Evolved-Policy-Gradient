import random

from epg.envs.envPicker.EnvironmentPicker import EnvironmentPicker
from epg.envs.environmentManager.EmptyMinigridEnvironmentManager import EmptyMinigridEnvironmentManager
from epg.envs.environmentManager.testEnvironmentManager.FourRoomsMinigridEnvironmentManager import \
    FourRoomsMinigridEnvironmentManager


class MiniGridEnvironmentPicker(EnvironmentPicker):

    def __init__(self, training_percentage=0.7):
        assert training_percentage < 1
        self.environmentManagersList = self.get_environment_managers(training_percentage)
        self.testEnvironmentManager = FourRoomsMinigridEnvironmentManager()

    def get_training_environment(self):
        return random.choice(self.environmentManagersList).get_training_environment()

    def get_validation_environment(self):
        return random.choice(self.environmentManagersList).get_validation_environment()

    def get_test_environment(self):
        return self.testEnvironmentManager.get_test_environment()

    def get_environment_managers(self, training_percentage):
        return [EmptyMinigridEnvironmentManager(training_percentage)]
