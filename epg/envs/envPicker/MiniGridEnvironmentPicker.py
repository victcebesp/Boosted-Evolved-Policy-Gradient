import random

from epg.envs.envPicker.EnvironmentPicker import EnvironmentPicker
from epg.envs.environmentManager.EmptyMinigridEnvironmentManager import EmptyMinigridEnvironmentManager


# MiniGrid environment picker. It returns a ExtendedEnvironment to use at train, validation and test time
class MiniGridEnvironmentPicker(EnvironmentPicker):

    def __init__(self, training_percentage=0.7):
        assert training_percentage < 1
        self.environmentManagersList = self.get_environment_managers(training_percentage)
        self.testEnvironmentManager = self.environmentManagersList[0]

    # Returns a randomly selected ExtendedEnvironment from a random EnvironmentManager to use at train time
    def get_training_environment(self):
        return random.choice(self.environmentManagersList).get_training_environment()

    # Returns a randomly selected ExtendedEnvironment from a random EnvironmentManager to use at validation time
    def get_validation_environment(self):
        return random.choice(self.environmentManagersList).get_validation_environment()

    # Returns a randomly selected ExtendedEnvironment from a random EnvironmentManager to use at test time
    def get_test_environment(self):
        return self.testEnvironmentManager.get_test_environment()

    # Returns a list of environment managers
    def get_environment_managers(self, training_percentage):
        return [EmptyMinigridEnvironmentManager(training_percentage)]
