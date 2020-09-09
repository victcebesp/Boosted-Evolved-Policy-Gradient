import numpy as np

from epg.evolutionSignalsCombinator.EvolutionSignalsCombinator import EvolutionSignalsCombinator
from epg.utils import relative_ranks

"""
    Calculates and returns the gradient penalizing or enhancing the explained in (Houthooft et al., 2018)
    with the normalized relative number of steps
"""


class RelativeTimeCombinator(EvolutionSignalsCombinator):

    def __init__(self):
        super().__init__()
        self.last_average_ep_length = 0

    """
        Calculates the new gradient penalizing or enhancing the former gradient
        using the normalized relative number of steps
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

        # Calculate the current normalized episode's length
        current_average_ep_length = average_validation_ep_length / optimal_validation_ep_length

        # Calculate the current normalize relative number of steps
        x = self.last_average_ep_length - current_average_ep_length

        # Shrink relative reward w.r.t. the number of episodes to the interval [-1, 1]
        beta = 1 - np.exp(-x) if x >= 0 else 1 - np.exp(x)

        # Combine former gradient with the shrunk transfer knowledge measure
        theta_grad = beta * (relative_ranks(returns).dot(noise) / outer_n_samples_per_ep - outer_l2 * theta)

        # Update last average episode's length
        self.last_average_ep_length = current_average_ep_length

        print('BETA:', beta)
        print('X:', x)

        return theta_grad

    # Returns the average episode's length for the current validation environment
    def get_average_ep_length(self, theta, env, objective):

        validation_results = []

        # Run inner-loop self.validation_samples times and save the results
        for i in range(self.validation_samples):
            validation_theta = theta[np.newaxis, :] + np.zeros((self.validation_samples, len(theta)))
            validation_results.append(objective(env, validation_theta[i], i))

        # Get episode's length
        episodes_average_length_array = np.asarray([np.mean(r['ep_length']) for r in validation_results])

        # Calculate average episode's length
        average_ep_length = episodes_average_length_array.mean()

        return average_ep_length
