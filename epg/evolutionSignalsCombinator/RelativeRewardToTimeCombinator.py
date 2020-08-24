import numpy as np

from epg.evolutionSignalsCombinator.EvolutionSignalsCombinator import EvolutionSignalsCombinator
from epg.utils import relative_ranks


class RelativeRewardToTimeCombinator(EvolutionSignalsCombinator):

    def __init__(self):
        super().__init__()
        self.last_reward_to_ep_length = 0

    def calculate_gradient(self, theta, noise, outer_n_samples_per_ep, outer_l2, NUM_EQUAL_NOISE_VECTORS, results_processed, env, objective):

        returns = np.asarray([r['returns'] for r in results_processed])

        average_reward_to_ep_length = self.get_average_reward_to_ep_length(theta, env, objective)
        x = average_reward_to_ep_length - self.last_reward_to_ep_length
        print('x:', x)

        beta = 1 - np.exp(-x * 4)
        print('BETA:', beta)

        noise = noise[::NUM_EQUAL_NOISE_VECTORS]
        returns = np.mean(returns.reshape(-1, NUM_EQUAL_NOISE_VECTORS), axis=1)

        theta_grad = relative_ranks(returns).dot(noise) * beta / outer_n_samples_per_ep - outer_l2 * theta

        return theta_grad

    def get_average_reward_to_ep_length(self, theta, env, objective):
        validation_results = []
        for i in range(self.validation_samples):
            validation_theta = theta[np.newaxis, :] + np.zeros((self.validation_samples, len(theta)))
            validation_results.append(objective(env, validation_theta[i], i))

        episodes_average_length_array = np.asarray([np.mean(r['ep_length']) for r in validation_results])
        episodes_average_reward_array = np.asarray([np.mean(r['ep_return']) for r in validation_results])
        rewards_to_ep_length = episodes_average_reward_array / episodes_average_length_array
        return rewards_to_ep_length.mean()
