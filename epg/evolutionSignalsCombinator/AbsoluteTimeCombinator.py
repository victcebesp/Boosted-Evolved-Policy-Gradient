import numpy as np

from epg.evolutionSignalsCombinator.EvolutionSignalsCombinator import EvolutionSignalsCombinator
from epg.utils import relative_ranks

"""
    Calculates and returns the gradient penalizing or enhancing the explained in (Houthooft et al., 2018)
    with the normalized absolute number of steps
"""


class AbsoluteTimeCombinator(EvolutionSignalsCombinator):

    """
        Calculates the new gradient penalizing or enhancing the former gradient
        using the normalized absolute number of steps
    """
    def calculate_gradient(self, theta, noise, outer_n_samples_per_ep, outer_l2, NUM_EQUAL_NOISE_VECTORS,
                           results_processed, env, objective):

        returns = np.asarray([r['returns'] for r in results_processed])

        # Calculate the average episode's length for the validation environment
        average_validation_ep_length = self.get_average_ep_length(theta, env, objective)

        noise = noise[::NUM_EQUAL_NOISE_VECTORS]
        returns = np.mean(returns.reshape(-1, NUM_EQUAL_NOISE_VECTORS), axis=1)

        # Get optimal episode's length for the received environment
        optimal_validation_ep_length = env.get_optimal_episode_length()

        # Calculate the normalized absolute number of steps
        x = -average_validation_ep_length / optimal_validation_ep_length

        # Shrink normalized absolute number of steps to the range [0, 1]
        beta = 1 - np.exp(1 + x)

        # Combine former gradient with the shrunk transfer knowledge measure
        theta_grad = beta * (relative_ranks(returns).dot(noise) / outer_n_samples_per_ep - outer_l2 * theta)

        print('BETA:', beta)
        print('X:', x)

        return theta_grad


    # Returns the average episode's length in the validation environment
    def get_average_ep_length(self, theta, env, objective):

        validation_results = []

        # Run inner-loop self.validation_samples times and save the results
        for i in range(self.validation_samples):
            validation_theta = theta[np.newaxis, :] + np.zeros((self.validation_samples, len(theta)))
            validation_results.append(objective(env, validation_theta[i], i))

        # Get episode lengths
        episodes_average_length_array = np.asarray([np.mean(r['ep_length']) for r in validation_results])

        # Calculate the average episode's length
        average_ep_length = episodes_average_length_array.mean()

        return average_ep_length
