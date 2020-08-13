class EnvironmentPicker:

    def __init__(self, env_id):
        self.env_id = env_id

    def get_training_environment(self):
        raise NotImplementedError

    def get_validation_environment(self):
        raise NotImplementedError
