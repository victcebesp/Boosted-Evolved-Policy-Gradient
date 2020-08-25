class EnvironmentPicker:

    def get_training_environment(self):
        raise NotImplementedError

    def get_validation_environment(self):
        raise NotImplementedError

    def get_test_environment(self):
        raise NotImplementedError
