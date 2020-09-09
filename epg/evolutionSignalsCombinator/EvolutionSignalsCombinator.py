"""
 Abstract class responsible for calculating the new gradient combining the one in (Houthooft et al., 2018) and a
# measure for the transfer of knowledge
"""
class EvolutionSignalsCombinator:

    def __init__(self, validation_samples=7):
        self.validation_samples = validation_samples

    # Returns the new gradient
    def calculate_gradient(self, theta, noise, outer_n_samples_per_ep, outer_l2, NUM_EQUAL_NOISE_VECTORS, results_processed, env, objective):
        raise NotImplementedError