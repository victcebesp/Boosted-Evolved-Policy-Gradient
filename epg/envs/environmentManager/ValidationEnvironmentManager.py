# Abstract class to provide to the environment picker validation environments
class ValidationEnvironmentManager:

    # Return a ExtendedEnvironment to use at validation time
    def get_validation_environment(self):
        raise NotImplementedError
