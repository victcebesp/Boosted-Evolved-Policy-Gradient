# Abstract class that provides to an ES ExtendedEnvironments to use at train, validation and test time
class EnvironmentPicker:

    # Return an ExtendedEnvironment to use at train time
    def get_training_environment(self):
        raise NotImplementedError

    # Return an ExtendedEnvironment to use at validation time
    def get_validation_environment(self):
        raise NotImplementedError

    # Return an ExtendedEnvironment to use at test time
    def get_test_environment(self):
        raise NotImplementedError
