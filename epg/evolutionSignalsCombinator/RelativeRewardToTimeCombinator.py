import numpy as np

from epg.evolutionSignalsCombinator.EvolutionSignalsCombinator import EvolutionSignalsCombinator
from epg.utils import relative_ranks

"""
    Calculates and returns the gradient penalizing or enhancing the explained in (Houthooft et al., 2018)
    with the relative reward w.r.t. the number of steps
"""


class RelativeRewardToTimeCombinator(EvolutionSignalsCombinator):

    def __init__(self):
        super().__init__()
        self.last_reward_to_ep_length = 0

    """
        Calculates the new gradient penalizing or enhancing the former gradient
        using the relative reward w.r.t. the number of steps
    """
    def calculate_gradient(self, theta, noise, outer_n_samples_per_ep, outer_l2, NUM_EQUAL_NOISE_VECTORS, results_processed, env, objective):

        returns = np.asarray([r['returns'] for r in results_processed])

        # Calculate the current average reward w.r.t. the current episode's length
        current_average_reward_to_ep_length = self.get_average_reward_to_ep_length(theta, env, objective)

        noise = noise[::NUM_EQUAL_NOISE_VECTORS]
        returns = np.mean(returns.reshape(-1, NUM_EQUAL_NOISE_VECTORS), axis=1)

        # Calculate the current relative reward w.r.t. the number of episodes
        x = current_average_reward_to_ep_length - self.last_reward_to_ep_length

        # Shrink relative reward w.r.t. the number of episodes to the interval [-1, 1]
        beta = 1 - np.exp(-x) if x >= 0 else 1 - np.exp(x)

        # Combine former gradient with the shrunk transfer knowledge measure
        theta_grad = beta * (relative_ranks(returns).dot(noise) / outer_n_samples_per_ep - outer_l2 * theta)

        # Update last reward w.r.t. the number of steps
        self.last_reward_to_ep_length = current_average_reward_to_ep_length

        print('x:', x)
        print('BETA:', beta)

        return theta_grad

    # Returns the average reward with respect to the number of steps
    def get_average_reward_to_ep_length(self, theta, env, objective):

        validation_results = []

        # Run inner-loop self.validation_samples times and save the results
        for i in range(self.validation_samples):
            validation_theta = theta[np.newaxis, :] + np.zeros((self.validation_samples, len(theta)))
            validation_results.append(objective(env, validation_theta[i], i))

        # Get episode's lengths
        episodes_average_length_array = np.asarray([np.mean(r['ep_length']) for r in validation_results])

        # Get episode's rewards
        episodes_average_reward_array = np.asarray([np.mean(r['ep_return']) for r in validation_results])

        # Calculate the average reward with respect to the episode's length
        rewards_to_ep_length = episodes_average_reward_array / episodes_average_length_array

        return rewards_to_ep_length.mean()
