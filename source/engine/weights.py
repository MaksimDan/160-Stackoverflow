import time
import logging
import progressbar
import numpy as np
import math
import itertools
from engine import Engine


class WeightVector:
    @staticmethod
    def cartisian_weight_approximation(n_features, axis_lim, inc):
        logging.info('Beginning cartisian_weight_approximation')
        t1 = time.time()

        temp_dict = []
        for weights in itertools.product(range(axis_lim[0], axis_lim[1]+inc, inc), repeat=n_features):
            engine = Engine()
            engine.rank_all_questions(weights, log_disabled=True)
            weight_dict = {f'f{i}'}
            temp_dict.append({'error': engine.residuals.get_total_error()})
        t2 = time.time()
        logging.info(f'\nFinished gradient descent in {(t2-t1)/60} minutes.'
                     f'\nThe final weight vector is {weights}.')

    @staticmethod
    def tune_weight_vector(n_features, base_alpha=2, exponential_increase=2):
        # start at from at least 1.5 for reasonably fast transition time
        # < 1.0 would make the make the weights decrease
        weights = np.random.rand(1, n_features)[0] + 1.5
        logging.info('Beginning tune_weight_vector')
        logging.info(f'Initial random weights: {weights}')

        t1 = time.time()
        engine = Engine()
        engine.rank_all_questions(weights, log_disabled=True)
        prev_error = engine.residuals.get_total_error()

        bar = progressbar.ProgressBar()
        for i in bar(range(len(weights))):
            logging.info(f'Now optimizing weight {i}.')
            weights[i] = WeightVector._tune_single_weight(weights, i, base_alpha, exponential_increase, prev_error)

        t2 = time.time()
        logging.info(f'\nFinished gradient descent in {(t2-t1)/60} minutes.'
                     f'\nThe final weight vector is {weights}.')
        engine.rank_all_questions(weights, log_disabled=False)
        return weights

    @staticmethod
    def _tune_single_weight(w, i, alpha, exp_increase, prev_error):
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
                e = Engine()
                e.rank_all_questions(w, log_disabled=True)
                p_error = c_error
                c_error = engine.residuals.get_total_error()
            return p_weight

        # first try increasing the weight
        prev_weight = w[i]
        alpha *= exp_increase
        adjust_weight(True)

        engine = Engine()
        engine.rank_all_questions(w, log_disabled=True)
        current_error = engine.residuals.get_total_error()
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

            engine = Engine()
            engine.rank_all_questions(w, log_disabled=True)
            current_error = engine.residuals.get_total_error()
            return loop_until_error_increase(current_error, prev_error, alpha, False)
