import matplotlib

from epg.evolution import ES

matplotlib.use('Agg')
import gym
import time

import numpy as np
from epg.launching import logger
from epg import utils
from epg import plotting
from epg.utils import PiecewiseSchedule

from epg.utils import Adam, relative_ranks
from epg.rollout import run_batch_rl

gym.logger.set_level(41)

# Statics
NUM_EQUAL_NOISE_VECTORS = 1
NUM_TEST_SAMPLES = 7


class SequentialES(ES):

    def train(self, outer_n_epoch, outer_l2, outer_std, outer_learning_rate, outer_n_samples_per_ep,
              n_cpu=None, fix_ppo=None, **_):

        if fix_ppo:
            ppo_factor_schedule = PiecewiseSchedule([(0, 1.), (int(outer_n_epoch / 16), 0.5)],
                                                    outside_value=0.5)
        else:
            ppo_factor_schedule = PiecewiseSchedule([(0, 1.), (int(outer_n_epoch / 8), 0.)],
                                                    outside_value=0.)

        outer_lr_scheduler = PiecewiseSchedule([(0, outer_learning_rate),
                                                (int(outer_n_epoch / 2), outer_learning_rate * 0.1)],
                                               outside_value=outer_learning_rate * 0.1)

        def objective(env, theta, pool_rank):
            agent = self.create_agent(env, pool_rank)
            loss_n_params = len(agent.get_loss().get_params_1d())
            agent.get_loss().set_params_1d(theta[:loss_n_params])
            if self._outer_evolve_policy_init:
                agent.pi.set_params_1d(theta[loss_n_params:])
            # Agent lifetime is inner_opt_freq * inner_max_n_epoch
            return run_batch_rl(env, agent,
                                inner_opt_freq=self._inner_opt_freq,
                                inner_buffer_size=self._inner_buffer_size,
                                inner_max_n_epoch=self._inner_max_n_epoch,
                                pool_rank=pool_rank,
                                ppo_factor=ppo_factor_schedule.value(epoch),
                                epoch=None)

        # Initialize theta.
        theta = self.init_theta(self._env)
        num_params = len(theta)
        logger.log('Theta dim: {}'.format(num_params))

        # Set up outer loop parameter update schedule.
        adam = Adam(shape=(num_params,), beta1=0., stepsize=outer_learning_rate, dtype=np.float32)

        begin_time, best_test_return = time.time(), -np.inf
        for epoch in range(outer_n_epoch):

            # Anneal outer learning rate
            adam.stepsize = outer_lr_scheduler.value(epoch)

            noise = np.random.randn(outer_n_samples_per_ep // NUM_EQUAL_NOISE_VECTORS, num_params)
            noise = np.repeat(noise, NUM_EQUAL_NOISE_VECTORS, axis=0)
            theta_noise = theta[np.newaxis, :] + noise * outer_std

            # Distributes theta_noise vectors to all nodes.
            logger.log('Running inner loops ...')

            start_time = time.time()

            results = []
            for i in range(outer_n_samples_per_ep):
                results.append(objective(self._env, theta_noise[i], i))

            # Extract relevant results
            returns = [utils.ret_to_obj(r['ep_final_rew']) for r in results]
            update_time = [np.mean(r['update_time']) for r in results]
            env_time = [np.mean(r['env_time']) for r in results]
            ep_length = [np.mean(r['ep_length']) for r in results]
            n_ep = [len(r['ep_length']) for r in results]
            mean_ep_kl = [np.mean(r['ep_kl']) for r in results]
            final_rets = [np.mean(r['ep_return'][-3:]) for r in results]

            results_processed_arr = np.asarray(
                [returns, update_time, env_time, ep_length, n_ep, mean_ep_kl, final_rets],
                dtype='float').ravel()

            # Do outer loop update
            end_time = time.time()
            logger.log(
                'All inner loops completed, returns gathered ({:.2f} sec).'.format(
                    time.time() - start_time))

            results_processed_arr = results_processed_arr.reshape(1, 7, outer_n_samples_per_ep)
            results_processed_arr = np.transpose(results_processed_arr, (0, 2, 1)).reshape(-1, 7)
            results_processed = [dict(returns=r[0],
                                      update_time=r[1],
                                      env_time=r[2],
                                      ep_length=r[3],
                                      n_ep=r[4],
                                      mean_ep_kl=r[5],
                                      final_rets=r[6]) for r in results_processed_arr]
            returns = np.asarray([r['returns'] for r in results_processed])

            # ES update
            noise = noise[::NUM_EQUAL_NOISE_VECTORS]
            returns = np.mean(returns.reshape(-1, NUM_EQUAL_NOISE_VECTORS), axis=1)
            theta_grad = relative_ranks(returns).dot(noise) / outer_n_samples_per_ep \
                         - outer_l2 * theta
            theta -= adam.step(theta_grad)

            # Perform `NUM_TEST_SAMPLES` evaluation runs on root 0.
            #if epoch % self._outer_plot_freq == 0 or epoch == outer_n_epoch - 1:
            #    start_test_time = time.time()
            #    logger.log('Performing {} test runs in parallel on node 0 ...'.format(NUM_TEST_SAMPLES))
            #    # Evaluation run with current theta
            #    test_results = pool.amap(
            #        objective,
            #        [self._env] * NUM_TEST_SAMPLES,
            #        theta[np.newaxis, :] + np.zeros((NUM_TEST_SAMPLES, num_params)),
            #        range(NUM_TEST_SAMPLES)
            #    ).get()
            #    plotting.plot_results(epoch, test_results)
            #    test_return = np.mean([utils.ret_to_obj(r['ep_return']) for r in test_results])
            #    if test_return > best_test_return:
            #        best_test_return = test_return
            #        # Save theta as numpy array.
            #        self.save_theta(theta)
            #    self.save_theta(theta, str(epoch))
            #    logger.log('Test runs performed ({:.2f} sec).'.format(time.time() - start_test_time))

            logger.logkv('Epoch', epoch)
            utils.log_misc_stats('Obj', logger, returns)
            logger.logkv('PPOFactor', ppo_factor_schedule.value(epoch))
            logger.logkv('EpochTimeSpent(s)', end_time - start_time)
            logger.logkv('TotalTimeSpent(s)', end_time - begin_time)
            #logger.logkv('BestTestObjMean', best_test_return)
            logger.dumpkvs()