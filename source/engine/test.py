from engine import Engine
from weights import WeightVector
from visuals import ResidualPlots
from visuals import DataUtilities
import math
import numpy as np
import pandas as pd
import logging
import os


class TestPlots:
    def __init__(self, n_features, weights):
        """
        objective: generate all plots
        :param n_features: int - number of features in the model
        :param weights: np.array - initialized weight vector
        """
        if weights is None:
            weights = np.repeat(1, n_features)
        self.engine = Engine(log_disabled=False, save_feature_matrices=False, visuals_active=True)
        self.engine.rank_all_questions(weights)

    def residual_matrix(self):
        ResidualPlots.plot_residual_matrix(self.engine.residuals.raw_residuals_per_question,
                                           'TEST_residual_matrix.png')

    def rank_distributions(self):
        ResidualPlots.plot_rank_error_distributions(self.engine.residuals.raw_residuals_per_question,
                                               'TEST_rank_distributions.png')

    def error_by_threshold(self):
        ResidualPlots.plot_rank_error_by_threshold(self.engine.residuals.raw_residuals_per_question,
                                              'TEST_error_by_threshold.png')

    def variance_per_rank(self):
        ResidualPlots.plot_entropy_per_rank(self.engine.recommender_user_matrix,
                                             'TEST_entropy_per_rank.png')

    def roc_curve_all_activities(self):
        score_matrix, label_matrix = self.engine.recommender_score_matrix, self.engine.recommender_label_matrix
        ResidualPlots.plot_roc_curve_for_all_activities(score_matrix, label_matrix, 'TEST_roc_curve_all_activities.png')

    @staticmethod
    def feature_weight_vs_error(inc):
        BASE_PATH = '../../../160-Stackoverflow-Data/'
        error_weight = pd.read_csv(BASE_PATH + 'residuals/data/error_by_linear_weight.csv')
        ResidualPlots.plot_weight_vs_error(error_weight, inc, 'TEST_feature_weight_vs_error.png')

    def distribution_loss_function(self, t):
        n_users = len(Engine.unique_users_list)
        threshold = math.floor(n_users * t)
        ResidualPlots.plot_loss_function_error_distribution(self.engine.residuals.get_loss_function_errors(threshold),
                                                            'TEST_distribution_loss_function.png')


class TestWeightVector:
    def __init__(self, feature_names):
        """
        objective: tests related to computations with
                the weight vector
        :param feature_names:
        """
        self.feature_names = feature_names

    def weight_tune(self):
        engine = Engine(visuals_active=False)
        w = WeightVector.tune_weight_vector(len(self.feature_names))
        engine.rank_all_questions(w)

    def linear_weight_opt(self, scaled_t, axis_lim, inc):
        n_users = len(Engine.unique_users_list)
        threshold = math.floor(n_users * scaled_t)
        WeightVector.linear_weight_correlate(self.feature_names, axis_lim, inc, threshold)


class Test:
    """
    objective: complete tests aggregate
    """
    @staticmethod
    def plot_tests(n_features, scaled_t, weights=None):
        pt = TestPlots(n_features, weights)
        pt.residual_matrix()
        pt.rank_distributions()
        pt.error_by_threshold()
        pt.roc_curve_all_activities()
        pt.variance_per_rank()
        pt.distribution_loss_function(scaled_t)

    @staticmethod
    def simplest_test(n_features):
        engine = Engine(visuals_active=False)
        weights = np.repeat(1, n_features)
        engine.rank_all_questions(weights)

    @staticmethod
    def save_residual_files(n_features, n_questions):
        engine = Engine(log_disabled=True, save_feature_matrices=True, visuals_active=False)
        weights = np.repeat(1, n_features)
        engine.rank_all_questions(weights)

        # residuals data frame
        flattened_full_residuals = \
            DataUtilities.flatten_full_residual_dictionary(engine.residuals.full_raw_residuals_per_question)
        pd.DataFrame(flattened_full_residuals).to_csv(f'residuals_{n_questions}_q.csv', index=False)

        # remaining additional files
        np.savetxt('label_matrix.csv', engine.recommender_label_matrix, delimiter=',')
        np.savetxt('score_matrix.csv', engine.recommender_score_matrix, delimiter=',')
        np.savetxt('user_matrix.csv', engine.recommender_user_matrix, delimiter=',')


def set_up_log_files(name):
    """
    objective: initialize log file
    :param name: string - name of log file
    :return:
    """
    # delete old log file, if existing
    try:
        os.remove(name)
    except OSError:
        pass

    # remove all handlers associated with the root logger object
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=name, level=logging.DEBUG)


if __name__ == '__main__':
    set_up_log_files('run.log')
    # Test.simplest_test(7)
    Test.plot_tests(n_features=11, scaled_t=.17, weights=np.array([50, 100, 130, 0, 75, 60, 0, 60, 75, 80, 20]))
    # Test.plot_tests(n_features=11, scaled_t=.17, weights=np.array([50, 100, 130, 0, 75, 60, 0, 60, 75, 80, 20]))

    # TestPlots.feature_weight_vs_error(6)

    # Test.save_residual_files(n_features=11, n_questions=800)

    # WTest = TestWeightVector(['user_avail', 'user_expertise_a', 'user_expertise_c', 'user_expertise_q',
    #                           'user_sim_expertise_a', 'user_sim_expertise_c', 'user_sim_expertise_q'])
    # WTest.linear_weight_opt(scaled_t=.15, axis_lim=(0, 550), inc=50)
    # plot_weight_vs_error(linear_error, 11, 'TEST_feature_weight_vs_error.png')

