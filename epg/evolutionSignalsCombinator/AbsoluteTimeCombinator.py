import numpy as np

from epg.evolutionSignalsCombinator.EvolutionSignalsCombinator import EvolutionSignalsCombinator
from epg.utils import relative_ranks


class AbsoluteTimeCombinator(EvolutionSignalsCombinator):

    def calculate_gradient(self, theta, noise, outer_n_samples_per_ep, outer_l2, NUM_EQUAL_NOISE_VECTORS, results_processed, env, objective):

        returns = np.asarray([r['returns'] for r in results_processed])

        average_validation_ep_length = self.get_average_ep_length(theta, env, objective)

        noise = noise[::NUM_EQUAL_NOISE_VECTORS]
        returns = np.mean(returns.reshape(-1, NUM_EQUAL_NOISE_VECTORS), axis=1)
        print('RETURNS:', returns)
        print('RELATIVE RANKS:', relative_ranks(returns))

        optimal_validation_ep_length = env.get_optimal_episode_length()
        #x = (optimal_ep_length - average_validation_ep_length) / average_validation_ep_length
        x = average_validation_ep_length / optimal_validation_ep_length
        #beta = 1 - np.exp(x)
        beta = 1 - np.exp(-x)

        theta_grad = beta * (relative_ranks(returns).dot(noise) / outer_n_samples_per_ep - outer_l2 * theta)

        print('BETA:', beta)
        print('X:', x)
        print('Optimal ep length:', optimal_validation_ep_length)
        print('Average validation ep length:', average_validation_ep_length)
        average_training_ep_length = np.asarray([r['ep_length'] for r in results_processed]).mean()
        print('Average training ep length:', average_training_ep_length)

        return theta_grad

    def get_average_ep_length(self, theta, env, objective):
        validation_results = []
        for i in range(self.validation_samples):
            validation_theta = theta[np.newaxis, :] + np.zeros((self.validation_samples, len(theta)))
            validation_results.append(objective(env, validation_theta[i], i))

        episodes_average_length_array = np.asarray([np.mean(r['ep_length']) for r in validation_results])
        average_ep_length = episodes_average_length_array.mean()
        return average_ep_length
