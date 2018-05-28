from engine import Engine
from weights import WeightVector
from visuals import ResidualPlots
import numpy as np
import logging
import os


class TestBasic:
    def __init__(self, n_features):
        self.n_features = n_features

    def random_weight(self):
        engine = Engine()
        weights = np.random.rand(1, self.n_features)[0] + 1.5
        engine.rank_all_questions(weights, None)


class TestPlots:
    def __init__(self, n_features):
        self.engine = Engine()
        weights = np.random.rand(1, n_features)[0] + 1.5
        self.engine.rank_all_questions(weights, None)

    def residual_matrix(self):
        ResidualPlots.plot_residual_matrix(self.engine.residuals.raw_residuals_per_question,
                                           'TEST_residual_matrix.png')

    def rank_distributions(self):
        ResidualPlots.plot_error_distributions(self.engine.residuals.raw_residuals_per_question,
                                               'TEST_rank_distributions.png')

    def error_by_threshold(self):
        ResidualPlots.plot_error_by_threshold(self.engine.residuals.raw_residuals_per_question,
                                              'TEST_error_by_threshold.png')

    def variance_per_rank(self):
        ResidualPlots.plot_variance_per_rank(self.engine.recommender_user_matrix,
                                             'TEST_entropy_per_rank.png')

    def ROC_curve_all_activities(self):
        score_matrix, label_matrix = self.engine.recommender_score_matrix, self.engine.recommender_label_matrix
        ResidualPlots.plot_roc_curve_for_all_activities(score_matrix, label_matrix, 'TEST_ROC_curve_all_activities.png')


class TestWeightVector:
    def __init__(self, n_features):
        self.n_features = n_features

    def weight_tune(self):
        engine = Engine()
        w = WeightVector.tune_weight_vector(self.n_features)
        print('Optimized weight vector:', w)
        engine.rank_all_questions(w, None)

    def build_error_matrix_by_cartisian_weight(self, axis_lim, inc):
        WeightVector.cartisian_weight_approximation(self.n_features, axis_lim, inc)


class Test:
    @staticmethod
    def plot_tests(n_features):
        pt = TestPlots(n_features)
        pt.residual_matrix()
        pt.rank_distributions()
        pt.error_by_threshold()
        pt.ROC_curve_all_activities()
        pt.variance_per_rank()

    @staticmethod
    def save_residual_files(n_features):
        engine = Engine()
        # weights = np.random.rand(1, n_features)[0] + 1.5
        weights = np.repeat(1, n_features)
        engine.rank_all_questions(weights, None, save_output=True)

        # residuals data frame
        raw_r = ResidualPlots.build_residual_dataframe(engine.residuals.raw_residuals_per_question)
        raw_r.to_csv('residuals_600_q.csv')


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
    Test.plot_tests(6)
    # Test.save_residual_files(6)

    # t = TestWeightVector(6)
    # t.build_error_matrix_by_cartisian_weight((-500, 1000), 500) # 52 hours
    # t.build_error_matrix_by_cartisian_weight((-250, 1000), 750) # 9 hours
