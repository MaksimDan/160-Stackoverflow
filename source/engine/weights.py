import time
import logging
import progressbar
import numpy as np
import math
import itertools
import pandas as pd
from engine import Engine


class WeightVector:
    @staticmethod
    def cartisian_weight_approximation(features, axis_lim, inc, t):
        """
        objective: brute force error optimization technique
        :param features: list(string) - feature names
        :param axis_lim: (int, int) - min and max limit to search per dim
        :param inc: int - increment from min to max
        :param t: int - threshold for loss function
        :return: pd.Dataframe - summary of results
        """
        # total_expected_run_time_in_hrs = (single_run_in_min *
        #                                   dimension_single_axis^dimension_weights)/60
        all_weights = list(itertools.product(range(axis_lim[0], axis_lim[1]+inc, inc), repeat=len(features)))
        total_engine_runs = len(all_weights)
        logging.info('Beginning cartisian_weight_approximation')
        logging.info(f'Planning out {total_engine_runs} engine runs.')

        t1 = time.time()
        results_dict = []
        bar = progressbar.ProgressBar()
        for weights in bar(all_weights):
            engine = Engine(log_disabled=True, visuals_active=False)
            engine.rank_all_questions(weights)
            weight_dict = {f: w for f, w in zip(features, weights)}
            weight_dict['rank_error'] = engine.residuals.get_total_rank_error()
            weight_dict['loss_function_error'] = engine.residuals.get_loss_function_total_error(t)
            results_dict.append(weight_dict)

        t2 = time.time()
        logging.info(f'cartisian_weight_approximation finished in {(t2-t1)/60} minutes.')
        pd.DataFrame(results_dict).to_csv('error_by_cartisian_weight.csv', index=False)

    @staticmethod
    def linear_weight_correlate(features, axis_lim, inc, t):
        """
        objective: a more naive error optimization technique, which
                   computes error linearly by keeping only a single
                   variable weight
        :param features: list(string) - feature names
        :param axis_lim: (int, int) - min and max limit to search per dim
        :param inc: int - increment from min to max
        :param t: int - threshold for loss function
        :return: pd.Dataframe - summary of results
        """
        results_dict = []
        init_w = np.repeat(1, len(features))

        logging.info('Beginning linear_weight_tune')
        total_engine_runs = len(list(range(axis_lim[0], axis_lim[1], inc))) * len(features)
        logging.info(f'Planning out {total_engine_runs} engine runs.')

        t1 = time.time()
        bar = progressbar.ProgressBar()
        for feature_i in bar(range(len(features))):
            for weight in range(axis_lim[0], axis_lim[1], inc):
                weights = init_w.copy()
                weights[feature_i] = weight

                # grab a random sample of questions per engine run
                engine = Engine(log_disabled=True, visuals_active=False)
                engine.X = engine.X_full.sample(30, random_state=42)
                engine.y = engine.y_full.sample(30, random_state=42)

                engine.rank_all_questions(weights)
                weight_dict = {feature: weights[i] for i, feature in enumerate(features)}
                weight_dict['rank_error'] = engine.residuals.get_total_rank_error()
                weight_dict['loss_function_error'] = engine.residuals.get_loss_function_total_error(t)
                results_dict.append(weight_dict)
        t2 = time.time()
        logging.info(f'linear_weight_tune finished in {(t2-t1)/60} minutes.')
        pd.DataFrame(results_dict).to_csv('error_by_linear_weight.csv', index=False)

    @staticmethod
    def tune_weight_vector(n_features, base_alpha=2, exponential_increase=5):
        """
        objective: another error optimization technique - this time built
                   to move forward in the direction of loss error
        :param n_features: int - number of features in model
        :param base_alpha: int - initial parameter to begin weight increase
        :param exponential_increase: int - value to exponentially increment base alpha to
        :return: optimized weights
        """
        # start at from at least 1.5 for reasonably fast transition time
        # < 1.0 would make the make the weights decrease
        weights = np.random.rand(1, n_features)[0] + 1.5
        logging.info('Beginning tune_weight_vector')
        logging.info(f'Initial random weights: {weights}')

        t1 = time.time()
        engine = Engine(log_disabled=True, visuals_active=False)
        engine.rank_all_questions(weights)
        prev_error = engine.residuals.get_total_rank_error()

        bar = progressbar.ProgressBar()
        for i in bar(range(len(weights))):
            logging.info(f'Now optimizing weight {i}.')
            weights[i] = WeightVector._tune_single_weight(weights, i, base_alpha, exponential_increase, prev_error)

        t2 = time.time()
        logging.info(f'\nFinished gradient descent in {(t2-t1)/60} minutes.'
                     f'\nThe final weight vector is {weights}.')
        return weights

    @staticmethod
    def _tune_single_weight(w, i, alpha, exp_increase, prev_error):
        """
        objective: helper function to tune_weight_vector to optimize a single error
        :param w: list(float) - weight vector
        :param i: int - index weight vector
        :param alpha: int - initial parameter to begin weight increase
        :param exp_increase: int - value to exponentially increment base alpha to
        :param prev_error: float - initial error
        :return:
        """
        def adjust_weight(increase):
            w[i] = math.pow(w[i], alpha if increase else 1 / alpha)

        def loop_until_error_increase(c_error, p_error, a, increase):
            p_weight = w[i]
            while c_error < p_error:
                print('why doe')
                logging.info(f'\tError decreased: w[{i}]={w[i]}, error={c_error}, alpha={a}')
                p_weight = w[i]

                a *= exp_increase
                adjust_weight(increase)
                e = Engine(log_disabled=True, visuals_active=False)
                e.rank_all_questions(w)
                p_error = c_error
                c_error = engine.residuals.get_total_rank_error()
            return p_weight

        # first try increasing the weight
        prev_weight = w[i]
        alpha *= exp_increase
        adjust_weight(True)

        engine = Engine(log_disabled=True, visuals_active=False)
        engine.rank_all_questions(w)
        current_error = engine.residuals.get_total_rank_error()
        # if the error decreases, repeat the same operation unit it does not
        if current_error < prev_error:
            print('error decrease with increased alpha')
            return loop_until_error_increase(current_error, prev_error, alpha, True)
        # otherwise restore the initial weight and error and
        # work backwards decreasing alpha
        else:
            print('error increased with increased alpha')
            w[i] = prev_weight
            adjust_weight(False)

            engine = Engine(log_disabled=True, visuals_active=False)
            engine.rank_all_questions(w)
            current_error = engine.residuals.get_total_rank_error()
            return loop_until_error_increase(current_error, prev_error, alpha, False)
