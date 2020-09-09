# Abstract class to provide to the environment picker test environments
class TestEnvironmentManager:

    # Return a ExtendedEnvironment to use at test time
    def get_test_environment(self):
        raise NotImplementedError
