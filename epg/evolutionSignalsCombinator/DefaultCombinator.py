from epg.evolutionSignalsCombinator.EvolutionSignalsCombinator import EvolutionSignalsCombinator
import numpy as np

from epg.utils import relative_ranks


class DefaultCombinator(EvolutionSignalsCombinator):

    def calculate_gradient(self, theta, noise, outer_n_samples_per_ep, outer_l2, NUM_EQUAL_NOISE_VECTORS, results_processed, env, objective):

        returns = np.asarray([r['returns'] for r in results_processed])

        noise = noise[::NUM_EQUAL_NOISE_VECTORS]
        returns = np.mean(returns.reshape(-1, NUM_EQUAL_NOISE_VECTORS), axis=1)
        theta_grad = relative_ranks(returns).dot(noise) / outer_n_samples_per_ep \
                     - outer_l2 * theta

        return theta_grad