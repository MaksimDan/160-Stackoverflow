import time
from copy import copy
import math
import os
import json
from collections import defaultdict
import pprint

# meta
import logging
import progressbar
import pickle

# data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

import sys
# in order for the pickle files to unpack
# the modules must be included where they were
# packed.
sys.path.append('../source/data/features')


class Feature:
    @staticmethod
    def load_p_file(p_file_path):
        try:
            with open(p_file_path, 'rb') as f:
                return pickle.load(f)
        except OSError as e:
            print(e)
            raise


class UserAvailability(Feature):
    pickle_path = '../../160-Stackoverflow-Data/train_test/engineered_features/user_availability_network.p'

    def __init__(self):
        self.UA_network = Feature.load_p_file(UserAvailability.pickle_path)

    def get_user_availability_probability(self, user_id, hour):
        try:
            user_aval = self.UA_network[int(user_id)]
        except KeyError as e:
            print(f'User {e} was unidentified in the user availability network.', e)
            raise
        try:
            return self.UA_network[int(user_id)][str(hour)]
        # if hour was not found, then there is no chance that the
        # user could have answered the question
        except KeyError:
            return 0


class UserExpertise(Feature):
    pickle_path = '../../160-Stackoverflow-Data/train_test/engineered_features/user_expertise_network.p'

    def __init__(self):
        self.UE_network = Feature.load_p_file(UserExpertise.pickle_path)

    def get_user_sum_expertise(self, user_id, tags):
        return sum([self._get_user_expertise(user_id, tag) for tag in tags])

    def _get_user_expertise(self, user_id, tag):
        # note that you are currently summing the frequency
        # of comments, questions, and answers. this inversely
        # will affect the residual analysis
        try:
            user_tag_expertise = self.UE_network[int(user_id)]
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
    pickle_path = '../../160-Stackoverflow-Data/train_test/engineered_features/user_basic_profile_network.p'

    def __init__(self):
        self.Users = Feature.load_p_file(BasicProfile.pickle_path)

    def get_all_measureable_features(self, user_id):
        return [self._get_user_reputation(user_id),
                self._get_views(user_id),
                self._get_upvotes(user_id),
                self._get_downvotes(user_id)]

    def _get_user_reputation(self, user_id):
        try:
            return self.Users[int(user_id)]['reputation']
        except KeyError as e:
            print(f'user_id {e} was not found.')
            raise

    def _get_creation_date(self, user_id):
        try:
            return self.Users[int(user_id)]['creation_date']
        except KeyError as e:
            print(f'user_id {e} was not found.')
            raise

    def _get_views(self, user_id):
        try:
            return self.Users[int(user_id)]['views']
        except KeyError as e:
            print(f'user_id {e} was not found.')
            raise

    def _get_upvotes(self, user_id):
        try:
            return self.Users[int(user_id)]['upvotes']
        except KeyError as e:
            print(f'user_id {e} was not found.')
            raise

    def _get_downvotes(self, user_id):
        try:
            return self.Users[int(user_id)]['downvotes']
        except KeyError as e:
            print(f'user_id {e} was not found.')
            raise


class TagNetwork(Feature):
    pickle_path = '../../160-Stackoverflow-Data/train_test/engineered_features/tag_network.p'

    def __init__(self):
        self.TAG_Network = Feature.load_p_file(TagNetwork.pickle_path)

    def shortest_path(self, a, b):
        pass


class Indicator:
    pickle_path = '../../160-Stackoverflow-Data/train_test/engineered_features/indicator_network.p'

    def __init__(self):
        self.I_Network = Feature.load_p_file(Indicator.pickle_path)


class ResidualColorMatrix:
    def __init__(self, color_map):
        self.color_map = color_map

    def build_residual_matrix(self, question_error, n_users):
        n_questions = len(question_error)
        residual_matrix = np.zeros((n_questions, n_users))
        for row, observed_i in question_error.items():
            for feature_type, users in observed_i.items():
                for u_id in users:
                    residual_matrix[row, u_id] = self.color_map[feature_type]
        return residual_matrix

    def colorize_residual_matrix(self, residual_matrix):
        pass


class Residuals:
    def __init__(self, X, y):
        self.X, self.y = X, y
        self.error = defaultdict(int)
        self.residual_matrix_raw = defaultdict(lambda: defaultdict(lambda: list()))

    def compute_and_store_residuals(self, score_matrix, y_index):
        observed = json.loads(self.y['owner_user_ids'].values[y_index])
        predicted = {int(score_matrix[i, 0]): {'index': i, 'score': score_matrix[i, -1]} for i in range(score_matrix.shape[0])}

        # user_id = observed['answerers'][0]
        # if user_id in predicted:
        #     print('yee')
        # else:
        #     print('wee')

        index_answer = [predicted[user]['index'] for user in observed['answerers']]
        index_comment = [predicted[user]['index'] for user in observed['commentors']]

        self.error['answer'] += sum(index_answer)
        self.error['comment'] += sum(index_comment)

        self.residual_matrix_raw[y_index]['i_answer'] = index_answer
        self.residual_matrix_raw[y_index]['i_comment'] = index_comment

    def get_total_error(self):
        return self.error['answer'] + self.error['comment']

    def get_summarize_statistics(self):
        stats = '\n Residual Summary Statistics\n'

        stats += 'Error per Individual User Activity'
        stats += (pprint.pformat(self.error, indent=4, width=1) + '\n')

        stats += 'Total Error\n'
        stats += str(self.get_total_error())
        return stats


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

        # load all users list (order does not matter)
        with open('users_list.p', 'rb') as fp:
            self.unique_users_list = pickle.load(fp)

        t2 = time.time()
        logging.info(f'Dataframe loading finished in time {t2 - t1} seconds.')

        self.all_residuals = Residuals(self.X, self.y)

    def build_and_display_residual_matrix(self, color_map):
        r_matrix = ResidualColorMatrix(color_map)
        r_matrix_raw = r_matrix.build_residual_matrix(self.all_residuals.residual_matrix_raw, len(self.unique_users_list))
        return r_matrix.colorize_residual_matrix(r_matrix_raw)

    def rank_all_questions(self, w):
        n_features = len(w)

        # n_features + 2, for the user id column and score column
        matrix_init = np.zeros(shape=(len(self.unique_users_list), n_features + 2), dtype=np.float)
        # add the users list as the first column of matrix
        matrix_init[:, 0] = np.array(self.unique_users_list)

        logging.info('Computing ranks...')
        t1 = time.time()

        # iterate through all questions
        bar = progressbar.ProgressBar()
        for index, row in bar(self.X.iterrows()):
            question_score_matrix = self._rank_question(row, copy(matrix_init), w)
            self.all_residuals.compute_and_store_residuals(question_score_matrix, index)

        t2 = time.time()
        logging.info(f'Ranking all questions computation finished in {(t2 - t1)/60} minutes.')
        logging.info(self.all_residuals.get_summarize_statistics())

    def _rank_question(self, X_row, M, w):
        for i, user in enumerate(M[:, 0]):
            M[i, 1:-1] = self._compute_feature_row_for_user(user, X_row)

        # weight each feature
        M[:, 1: -1] = np.multiply(M[:, 1:-1], w)

        # now transform to fix units (excluding users column, and the score column)
        scaler = MinMaxScaler()
        M[:, 1:-1] = scaler.fit_transform(M[:, 1:-1])

        # compute the final score based off the features by summing
        # all the rows (excluding the user column and the score column)
        M[:, -1] = np.array([np.sum(M[i, 1:-1]) for i in range(M.shape[0])])

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
    @staticmethod
    def tune_weight_vector(n_features, base_alpha=2, exponential_increase=2):
        # start at from at least 1.5 for reasonably fast transition time
        # < 1.0 would make the make the weights decrease
        weights = np.random.rand(1, n_features)[0] + 1.5
        logging.info('Beginning gradient descent...')

        engine = Engine()
        engine.rank_all_questions(weights)
        error = engine.all_residuals.get_total_error()

        bar = progressbar.ProgressBar()
        for i in bar(range(len(weights))):
            weights[i] = WeightVector._tune_single_weight(weights, i, base_alpha, exponential_increase, error)
        logging.info(f'Finished gradient descent. The final weight vector is {weights}.')

        return weights

    @staticmethod
    def _tune_single_weight(w, i, alpha, exp_increase, prev_error):
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
            error = engine_trail.all_residuals.get_total_error()

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


class Test:
    def __init__(self, n_features):
        self.n_features = n_features

    def random_weight(self):
        engine = Engine()
        weights = np.random.rand(1, self.n_features)[0] + 1.5
        engine.rank_all_questions(weights)

    def random_weight_with_visual_residual(self, color_map):
        engine = Engine()
        weights = np.random.rand(1, self.n_features)[0] + 1.5
        engine.rank_all_questions(weights)
        engine.build_and_display_residual_matrix(color_map)

    def weight_tune(self):
        engine = Engine()
        w = WeightVector.tune_weight_vector(self.n_features)
        engine.rank_all_questions(w)

    def weight_tune_with_visual_residual(self, color_map):
        engine = Engine()
        w = WeightVector.tune_weight_vector(self.n_features)
        engine.rank_all_questions(w)
        engine.build_and_display_residual_matrix(color_map)


# Feature Summary:
#   BasicProfile (4):
#     - reputation
#     - views
#     - up votes
#     - down votes
#   UserExpertise (1) ignored for now
#   UserAvailability (1)


if __name__ == '__main__':
    # delete old log file, if existing
    try:
        os.remove('engine_log.log')
    except OSError:
        pass

    logging.basicConfig(filename='engine_log.log', level=logging.INFO)

    colormap = {'i_answer': {'label': 1, 'rgb': (0, 0, 1)},
                'i_comment': {'label': 2, 'rgb': (0, 1, 0)} }

    test = Test(5)
    test.random_weight()
    # test.random_weight_with_visual_residual(colormap)
    # test.weight_tune()
    # test.weight_tune_with_visual_residual(colormap)
