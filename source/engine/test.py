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
        engine.rank_all_questions(weights)


class TestPlots:
    def __init__(self, n_features):
        self.engine = Engine()
        weights = np.random.rand(1, n_features)[0] + 1.5
        self.engine.rank_all_questions(weights)
        raw_r = ResidualPlots._build_residual_dataframe(self.engine.residuals.raw_residuals_per_question)
        raw_r.to_csv('residuals_300_questions.csv')

    def residual_matrix(self):
        ResidualPlots.plot_residual_matrix(self.engine.residuals.raw_residuals_per_question,
                                           'TEST_residual_matrix.png')

    def rank_distributions(self):
        ResidualPlots.plot_error_distributions(self.engine.residuals.raw_residuals_per_question,
                                               'TEST_rank_distributions.png')

    def error_by_threshold(self):
        ResidualPlots.plot_error_by_threshold(self.engine.residuals.raw_residuals_per_question,
                                              'TEST_error_by_threshold.png')


class TestWeightVector:
    def __init__(self, n_features):
        self.n_features = n_features

    def weight_tune(self):
        engine = Engine()
        w = WeightVector.tune_weight_vector(self.n_features)
        print('Optimized weight vector:', w)
        engine.rank_all_questions(w)

    def build_error_matrix_by_cartisian_weight(self):
        pass


class Test:
    def __init__(self, n_features):
        self.t = TestBasic(n_features)
        self.pt = TestPlots(n_features)
        self.wt = TestWeightVector(n_features)

    def test_all(self):
        self.basic_tests()
        self.plot_tests()
        self.weight_vector_tests()

    def basic_tests(self):
        self.t.random_weight()

    def plot_tests(self):
        self.pt.residual_matrix()
        self.pt.rank_distributions()
        self.pt.error_by_threshold()

    def weight_vector_tests(self):
        self.wt.weight_tune()
        self.wt.build_error_matrix_by_cartisian_weight()


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
  UserExpertise (1) ignored for now
  UserAvailability (1)
'''

if __name__ == '__main__':
    set_up_log_files('engine_info.log')
    t = Test(5)
    # t.basic_tests()
    t.plot_tests()