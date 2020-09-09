#  Abstract class responsible for extending MiniGrid environments with the get_optimal_episodes_length method
class ExtendedEnvironment:

    # Returns the environment's optimal length or an approximation
    def get_optimal_episode_length(self):
        raise NotImplementedError
