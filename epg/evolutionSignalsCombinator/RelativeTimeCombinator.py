import numpy as np

from epg.evolutionSignalsCombinator.EvolutionSignalsCombinator import EvolutionSignalsCombinator
from epg.utils import relative_ranks


class RelativeTimeCombinator(EvolutionSignalsCombinator):

    def __init__(self):
        self.last_average_ep_length = 0

    def calculate_gradient(self, theta, noise, outer_n_samples_per_ep, outer_l2, NUM_EQUAL_NOISE_VECTORS,
                           results_processed):
        returns = np.asarray([r['returns'] for r in results_processed])
        average_ep_length = np.asarray([r['ep_length'] for r in results_processed]).mean()
        relative_ep_length = self.last_average_ep_length - average_ep_length
        self.last_average_ep_length = average_ep_length

        noise = noise[::NUM_EQUAL_NOISE_VECTORS]
        returns = np.mean(returns.reshape(-1, NUM_EQUAL_NOISE_VECTORS), axis=1)
        relative_ranks_absolute_time = relative_ranks(returns) / relative_ep_length
        theta_grad = relative_ranks_absolute_time.dot(noise) / outer_n_samples_per_ep \
                     - outer_l2 * theta

        return theta_grad