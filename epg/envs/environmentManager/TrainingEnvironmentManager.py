# Abstract class to provide to the environment picker train environments
class TrainingEnvironmentManager:

    # Return a ExtendedEnvironment to use at train time
    def get_training_environment(self):
        raise NotImplementedError
