from engine import Engine
from weights import WeightVector
from visuals import ResidualPlots
import math
import numpy as np
import pandas as pd
import logging
import os


class TestPlots:
    def __init__(self, n_features, weights=None):
        if not weights:
            weights = np.repeat(1, n_features)
        self.engine = Engine()
        self.engine.rank_all_questions(weights, None)

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
        error_weight = pd.read_csv(BASE_PATH + 'residuals/error_by_linear_weight.csv')
        feature_order = ['availability', 'reputation', 'views', 'upvotes', 'downvotes', 'expertise']
        ResidualPlots.plot_weight_vs_error(error_weight, feature_order, inc, 'TEST_feature_weight_vs_error.png')

    def distribution_loss_function(self, t):
        n_users = len(Engine.unique_users_list)
        threshold = math.floor(n_users * t)
        ResidualPlots.plot_loss_function_error_distribution(self.engine.residuals.get_loss_function_errors(threshold),
                                                            'TEST_distribution_loss_function.png')


class TestWeightVector:
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def weight_tune(self):
        engine = Engine(visuals_active=False)
        w = WeightVector.tune_weight_vector(len(self.feature_names))
        engine.rank_all_questions(w, None)

    def brute_force_weight_optimization(self, axis_lim, inc):
        WeightVector.cartisian_weight_approximation(self.feature_names, axis_lim, inc)

    def linear_weight_optimization(self, axis_lim, inc):
        WeightVector.linear_weight_tune(self.feature_names, axis_lim, inc)


class Test:
    @staticmethod
    def plot_tests(n_features, inc, t):
        pt = TestPlots(n_features)
        pt.residual_matrix()
        pt.rank_distributions()
        pt.error_by_threshold()
        pt.roc_curve_all_activities()
        pt.variance_per_rank()
        TestPlots.feature_weight_vs_error(inc)
        pt.distribution_loss_function(t)

    @staticmethod
    def save_residual_files(n_features, n_questions):
        engine = Engine(save_feature_matrices=True, visuals_active=False)
        weights = np.repeat(1, n_features)
        engine.rank_all_questions(weights, None)

        # residuals data frame
        raw_r = ResidualPlots.build_residual_dataframe(engine.residuals.raw_residuals_per_question)
        raw_r.to_csv(f'residuals_{n_questions}_q.csv')

    @staticmethod
    def simplest_test(n_features):
        engine = Engine(visuals_active=False)
        weights = np.repeat(1, n_features)
        engine.rank_all_questions(weights, None)


def set_up_log_files(name):
    # delete old log file, if existing
    try:
        os.remove(name)
    except OSError:
        pass

    # remove all handlers associated with the root logger object
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=name, level=logging.DEBUG)


'''
Current Feature Summary:
    BasicProfile (4):
        - reputation
        - views
        - up votes
        - down votes
    UserExpertise (1) 
    UserAvailability (1)

Remaining Features:
    Indicator Network
    
Post Features and Residual Analysis:
    Tag Network
    User Similarity Network
    Similiar Questions Network
    PostLinks
'''

if __name__ == '__main__':
    set_up_log_files('run.log')
    # Test.simplest_test(7)

    # TestPlots.feature_weight_vs_error(6)

    Test.plot_tests(7, 6, .17)
    # Test.save_residual_files(7)

    # features = ['availability', 'reputation', 'views', 'upvotes', 'downvotes', 'expertise', 'tag_sim_expertise']
    # t = TestWeightVector(features)
    # t.linear_weight_optimization((-200, 1000), 200)
    # t.build_error_matrix_by_cartisian_weight((-500, 1000), 500) # 52 hours
    # t.build_error_matrix_by_cartisian_weight((-250, 1000), 750) # 9 hours
