import json
import time
from copy import copy
import math
import os

# meta
import logging
import progressbar
import pickle

# data
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class Feature:
    pass


class UserAvailability(Feature):
    JSON_PATH = '../../160-Stackoverflow-Data/train_test/engineered_features/user_availability_network.json'

    def __init__(self):
        try:
            with open(UserAvailability.JSON_PATH) as f:
                self.UA_network = json.load(f)
        except OSError as e:
            print(f'{UserAvailability.JSON_PATH} was not found.', e)
            raise

    def get_user_availability_probability(self, user_id, hour):
        try:
            user_aval = self.UA_network[str(int(user_id))]
        except KeyError as e:
            print(f'User {e} was unidentified in the user availability network.', e)
            raise
        try:
            return self.UA_network[str(int(user_id))][str(hour)]
        # if hour was not found, then there is no chance that the
        # user could have answered the question
        except KeyError:
            return 0


class UserExpertise(Feature):
    JSON_PATH = '../../160-Stackoverflow-Data/train_test/engineered_features/user_expertise_network.json'

    def __init__(self):
        try:
            with open(UserExpertise.JSON_PATH) as f:
                self.UE_network = json.load(f)
        except OSError as e:
            print('Warning: User Expertise file was not found.', e)

    def get_user_sum_expertise(self, user_id, tags):
        return sum([self._get_user_expertise(user_id, tag) for tag in tags])

    def _get_user_expertise(self, user_id, tag):
        # note that you are currently summing the frequency
        # of comments, questions, and answers. this inversely
        # will affect the residual analysis
        try:
            user_tag_expertise = self.UE_network[str(user_id)]
            if tag not in user_tag_expertise:
                return 0
            else:
                # otherwise return the sum of the frequencies of posts per tag
                a = user_tag_expertise[tag].get('n_answers', 0.0)
                b = user_tag_expertise[tag].get('n_comments', 0.0)
                c = user_tag_expertise[tag].get('n_questions', 0.0)
                return a + b + c
        except KeyError as e:
            print(f'user_id {user_id} was not found.', e)
            raise


class BasicProfile(Feature):
    JSON_PATH = '../../160-Stackoverflow-Data/train_test/engineered_features/user_basic_profile_network.json'

    def __init__(self):
        try:
            with open(BasicProfile.JSON_PATH) as f:
                self.Users = json.load(f)
        except OSError as e:
            print(f'{BasicProfile.JSON_PATH} was not found.', e)
            raise

    def get_all_measureable_features(self, user_id):
        return [self._get_user_reputation(user_id),
                self._get_views(user_id),
                self._get_upvotes(user_id),
                self._get_downvotes(user_id)]

    def _get_user_reputation(self, user_id):
        try:
            return self.Users[str(int(user_id))]['reputation']
        except KeyError as e:
            print(f'user_id {e} was not found.')
            raise

    def _get_creation_date(self, user_id):
        try:
            return self.Users[str(int(user_id))]['creation_date']
        except KeyError as e:
            print(f'user_id {e} was not found.')
            raise

    def _get_views(self, user_id):
        try:
            return self.Users[str(int(user_id))]['views']
        except KeyError as e:
            print(f'user_id {e} was not found.')
            raise

    def _get_upvotes(self, user_id):
        try:
            return self.Users[str(int(user_id))]['upvotes']
        except KeyError as e:
            print(f'user_id {e} was not found.')
            raise

    def _get_downvotes(self, user_id):
        try:
            return self.Users[str(int(user_id))]['downvotes']
        except KeyError as e:
            print(f'user_id {e} was not found.')
            raise


class TagNetwork(Feature):
    JSON_PATH = '../../160-Stackoverflow-Data/train_test/engineered_features/tag_network.json'

    def __init__(self):
        try:
            with open(TagNetwork.JSON_PATH) as f:
                self.TAG_Network = json.load(f)
        except OSError as e:
            print(f'{TagNetwork.JSON_PATH} was not found.', e)
            raise

    def shortest_path(self, a, b):
        pass


class Indicator:
    def __init__(self):
        pass


class Residuals:
    def __init__(self, X, y):
        self.recommender_ranks = []
        self.observed_ranks = []
        self.user_rank_intersection = []
        self.total_error = 0

    def compute_and_store_residuals(self, score_matrix, y_index):
        pass

    def build_residual_matrix(self):
        pass


class Engine:
    def __init__(self):
        logging.info('Loading all data frames...')
        t1 = time.time()

        # load questions and all user activities
        self.X = pd.read_csv('../../160-Stackoverflow-Data/train_test/X100.csv')
        self.y = pd.read_csv('../../160-Stackoverflow-Data/train_test/y100.csv')
        self.X['CreationDate'] = pd.to_datetime(self.X['CreationDate'], format="%Y-%m-%dT%H:%M:%S")

        # load engineered features
        self.user_availability = UserAvailability()
        self.user_profile = BasicProfile()
        # todo: ignore these for now, integrate them later once everything works
        # self.tag_network = TagNetwork()
        # self.user_expertise = UserExpertise()

        t2 = time.time()
        logging.info(f'Dataframe loading finished in time {t2 - t1} seconds.')

        self.all_residuals = Residuals(self.X, self.y)

    def rank_all_questions(self, w):
        n_features = len(w)

        # all the users (order does not matter)
        with open('users_list.p', 'rb') as fp:
            unique_users_list = pickle.load(fp)
            # n_features + 2, for the user id column and score column
            matrix_init = np.zeros(shape=(len(unique_users_list), n_features + 2), dtype=np.float)
            # add the users list as the first column of matrix
            matrix_init[:, 0] = np.array(unique_users_list)

        logging.info('Computing ranks...')
        t1 = time.time()

        # iterate through all questions
        bar = progressbar.ProgressBar()
        for index, row in bar(self.X.iterrows()):
            question_score_matrix = self._rank_question(row, copy(matrix_init), w)
            self.all_residuals.compute_and_store_residuals(question_score_matrix, index)

        t2 = time.time()
        logging.info(f'Ranking all questions computation finished in {(t2 - t1)/60} minutes.')

    def _rank_question(self, X_row, M, w):
        for i, user in enumerate(M[:, 0]):
            M[i, 1:-1] = self._compute_feature_row_for_user(user, X_row)

        # now transform to fix units (excluding users column, and the score column)
        scaler = StandardScaler()
        scaler.fit_transform(M[:, 1:-1])

        # weight each feature
        M[:, 1: -1] = M[:, 1:-1]*w

        # compute the final score based off the features by summing
        # all the rows (excluding the user column and the score column)
        M[:, -1] = [sum(np.sum(M[i, 1:-1])) for i in range(M.shape[0])]

        # by default sort will sort by the last column
        return Engine._sort_matrix_by_column(M, -1, ascending=False)

    def _compute_feature_row_for_user(self, user_id, X_row):
        user_avail = self.user_availability.get_user_availability_probability(user_id, X_row.CreationDate.hour)
        # user_expertise = self.user_expertise.get_user_sum_expertise(user_id, X_row['Tags'].split())
        user_basic_profile = self.user_profile.get_all_measureable_features(user_id)
        # [user_expertise]
        return [user_avail] + user_basic_profile

    @staticmethod
    def _sort_matrix_by_column(M, i_column, ascending=True):
        multiplier = 1 if ascending else -1
        return M[np.argsort(multiplier*M[:, i_column])]


class WeightVector:
    def __init__(self, n_features):
        self.n_features = n_features

    def tune_weight_vector(self, base_alpha, exponential_increase=2):
        # start at from at least 1.5 for reasonably fast transition time
        # < 1.0 would make the make the weights decrease
        weights = np.random.rand(1, self.n_features)[0] + 1.5
        logging.info('Beginning gradient descent...')

        engine = Engine()
        engine.rank_all_questions(weights)
        error = engine.all_residuals.total_error

        bar = progressbar.ProgressBar()
        for i in bar(range(len(weights))):
            self._tune_single_weight(weights, i, base_alpha, exponential_increase, error)
        logging.info(f'Finished gradient descent. The final weight vector is {weights}.')

    def _tune_single_weight(self, w, i, alpha, exp_increase, prev_error):
        def adjust_weight(increase):
            w[i] = math.pow(w[i], alpha if increase else 1 / alpha)

        break_state = 4
        sm = [{},
              {'error_decrease': {'next_state': 2, 'op': adjust_weight, 'param': True},
               'error_increase': {'next_state': 3, 'op': adjust_weight, 'param': False}},

              {'error_decrease': {'next_state': 2, 'op': adjust_weight, 'param': True},
               'error_increase': {'next_state': break_state, 'op': None}},

              {'error_decrease': {'next_state': 3, 'op': adjust_weight, 'param': False},
               'error_increase': {'next_state': break_state, 'op': None}}]

        prev_weight = w[i]
        current_state = 1
        while current_state != break_state:
            engine_trail = Engine()
            engine_trail.rank_all_questions(w)
            error = engine_trail.all_residuals.total_error

            # if the error is greater, move in the reverse direction
            if error < prev_error:
                # adjust the weight accordingly
                sm[current_state]['op'](sm[current_state]['param'])
                prev_weight = w[i]
                current_state = sm[current_state]['error_decrease']
                alpha *= exp_increase
                logging.info(f'Error decreased: w[{i}]={w[i]}, error={error}')
            else:
                if sm[current_state]['op']:
                    sm[current_state]['op'](sm[current_state]['param'])
                current_state = sm[current_state]['error_increase']
                logging.info(f'Error increased: w[{i}]={w[i]}, error={error}')
            prev_error = error

        return prev_weight


def basic_test():
    # delete old log file, if existing
    try:
        os.remove('engine_log.log')
    except OSError:
        pass

    logging.basicConfig(filename='engine_log.log', level=logging.INFO)
    engine = Engine()

    # current features:
    #   BasicProfile (4):
    #     - reputation
    #     - views
    #     - up votes
    #     - down votes
    #   UserExpertise (1) ignored for now
    #   UserAvailability (1)

    # something random for now
    n_features = 5
    weights = np.random.rand(1, n_features)[0] + 1.5
    engine.rank_all_questions(weights)


if __name__ == '__main__':
    basic_test()
