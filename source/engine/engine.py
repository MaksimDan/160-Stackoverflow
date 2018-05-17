import time
from copy import copy
import math
import os
import json
from collections import defaultdict

# visualization
import matplotlib.pyplot as plt
import matplotlib.colors
import pprint


# meta
import logging
import progressbar
import pickle
import sys

# data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# preprocessing
# in order for the pickle files to unpack
# the modules must be included where they were
# packed.
sys.path.append('../source/data/features')
BASE_PATH = '../../160-Stackoverflow-Data/train_test/'
progressbar.streams.flush()


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
    pickle_path = BASE_PATH + 'engineered_features/user_availability_network.p'

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
    pickle_path = BASE_PATH + 'engineered_features/user_expertise_network.p'

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
    pickle_path = BASE_PATH + 'engineered_features/user_basic_profile_network.p'
    n_features = 4

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
    pickle_path = BASE_PATH + 'engineered_features/tag_network.p'

    def __init__(self):
        self.TAG_Network = Feature.load_p_file(TagNetwork.pickle_path)

    def shortest_path(self, a, b):
        pass


class Indicator:
    pickle_path = BASE_PATH + 'engineered_features/indicator_network.p'

    def __init__(self):
        self.I_Network = Feature.load_p_file(Indicator.pickle_path)


class ResidualColorMatrix:
    rbg = {'GREEN': (0, 255, 0), 'CYAN': (0, 255, 255), 'BLUE': (0, 128, 255), 'PURPLE': (127, 0, 255),
           'LIGHT_GREY': (224, 224, 224), 'ORANGE': (255, 128, 0), 'YELLOW': (255, 255, 0), 'RED': (255, 0, 0)}
    color_map = {'base': {'label': 0, 'rgb': rbg['LIGHT_GREY']},
                 'i_answer': {'label': 1, 'rgb': rbg['BLUE']},
                 'i_comment': {'label': 2, 'rgb': rbg['CYAN']},
                 'i_upvote': {'label': 3, 'rgb': rbg['GREEN']},
                 'i_downvote': {'label': 4, 'rgb': rbg['RED']},
                 'i_favorite': {'label': 5, 'rgb': rbg['YELLOW']},
                 'i_edit': {'label': 6, 'rgb': rbg['PURPLE']}}

    @staticmethod
    def build_residual_matrix(question_error, n_users):
        n_questions = len(question_error)
        residual_matrix = np.zeros((n_questions, n_users), dtype=np.int)
        for row, observed_i in question_error.items():
            for feature_type, rank in observed_i.items():
                for i in rank:
                    residual_matrix[row, i] = ResidualColorMatrix.color_map[feature_type]['label']
        return residual_matrix

    @staticmethod
    def colorize_residual_matrix(mat, save_path):
        ca = [list(ResidualColorMatrix.color_map[key]['rgb']) for key in ResidualColorMatrix.color_map.keys()]
        colors = np.matrix(ca)/255
        cmap = matplotlib.colors.ListedColormap(colors)
        norm = matplotlib.colors.BoundaryNorm(np.arange(len(ca)+1)-0.5, len(ca))

        plt.imshow(mat, cmap=cmap, norm=norm)
        plt.axis('off')
        cb = plt.colorbar(ticks=np.arange(len(ca)))
        cb.ax.set_yticklabels(ResidualColorMatrix.color_map.keys())
        plt.title('Residual Matrix')
        plt.savefig(save_path)
        plt.show()


class Residuals:
    def __init__(self, X, y):
        self.X, self.y = X, y
        self.error = defaultdict(int)
        self.residual_matrix_raw = defaultdict(lambda: defaultdict(lambda: list()))

    def compute_and_store_residuals(self, score_matrix, y_index):
        observed = json.loads(self.y['owner_user_ids'].values[y_index])
        predicted = {int(score_matrix[i, 0]): {'index': i, 'score': score_matrix[i, -1]} for i in range(score_matrix.shape[0])}

        # TODO: ensure all users have proper indexing
        # index_answer = [predicted[user]['index'] for user in observed['answerers']]
        # index_comment = [predicted[user]['index'] for user in observed['commentors']]

        index_answer, index_comment = [], []
        for user in observed['answerers']:
            try:
                index_answer.append(predicted[user]['index'])
            except KeyError:
                logging.debug(f'Missing answerer user: {user}')

        for user in observed['commentors']:
            try:
                index_comment.append(predicted[user]['index'])
            except KeyError:
                logging.debug(f'Missing commenter user: {user}')

        self.error['answer'] += sum(index_answer)
        self.error['comment'] += sum(index_comment)

        self.residual_matrix_raw[y_index]['i_answer'] = index_answer
        self.residual_matrix_raw[y_index]['i_comment'] = index_comment

    def get_total_error(self):
        return self.error['answer'] + self.error['comment']

    def get_summarize_statistics(self, n_total_users):
        stats = 'Residual Summary Statistics\n\n'

        stats += 'Error per Individual User Activity:\n'
        stats += (pprint.pformat(self.error, indent=4, width=1) + '\n')

        stats += 'Total Error: '
        stats += str(self.get_total_error()) + '\n'

        n_questions = len(self.residual_matrix_raw)
        stats += 'Average Rank Error Per Question: '
        stats += str(self.get_total_error()/n_questions) + '\n'

        stats += 'Average Percent Rank Error Per Question: '
        stats += str(self.get_total_error()/n_questions/n_total_users) + '\n'

        # todo: average percent tank error broken down by type of response
        return stats


class Engine:
    logging.info('Loading all data frames...')
    t1 = time.time()

    # load questions and all user activities
    X = pd.read_csv(BASE_PATH + 'X100.csv').head(10)
    y = pd.read_csv(BASE_PATH + 'y100.csv').head(10)
    X['CreationDate'] = pd.to_datetime(X['CreationDate'], format="%Y-%m-%dT%H:%M:%S")

    # load engineered features
    user_availability = UserAvailability()
    user_profile = BasicProfile()
    # todo: ignore these for now, integrate them later once everything works
    # tag_network = TagNetwork()
    # user_expertise = UserExpertise()

    # load all users list (order does not matter)
    with open(BASE_PATH + 'meta/users_list.p', 'rb') as fp:
        unique_users_list = pickle.load(fp)

    t2 = time.time()
    logging.info(f'Dataframe loading finished in time {t2 - t1} seconds.\n')

    def __init__(self):
        self.all_residuals = Residuals(Engine.X, Engine.y)

    def build_and_display_residual_matrix(self, save_path):
        r_matrix = ResidualColorMatrix()
        r_matrix_raw = r_matrix.build_residual_matrix(self.all_residuals.residual_matrix_raw, len(Engine.unique_users_list))
        r_matrix.colorize_residual_matrix(r_matrix_raw, save_path)

    def rank_all_questions(self, w, log_disabled=False):
        print('\n Ranking All Questions \n')
        logger = logging.getLogger()
        logger.disabled = log_disabled
        n_features = len(w)

        # n_features + 2, for the user id column and score column
        matrix_init = np.zeros(shape=(len(Engine.unique_users_list), n_features + 2), dtype=np.float)
        # add the users list as the first column of matrix
        matrix_init[:, 0] = np.array(Engine.unique_users_list)

        logging.info(f'Computing ranks using weight vector: {w}')
        t1 = time.time()

        # iterate through all questions
        bar = progressbar.ProgressBar()
        print(w)
        w_new = ','.join(str(x)[0:3] for x in w)
        for index, row in bar(Engine.X.iterrows()):
            question_score_matrix = Engine._rank_question(row, copy(matrix_init), w)
            # np.savetxt(f'r_matrix_q-{index}_w-{w_new}.csv', question_score_matrix[:, 0], delimiter=",")
            self.all_residuals.compute_and_store_residuals(question_score_matrix, index)

        t2 = time.time()
        logging.info(f'Ranking all questions computation finished in {(t2 - t1)/60} minutes.')
        logging.info(self.all_residuals.get_summarize_statistics(len(Engine.unique_users_list)))
        logger.disabled = False

    @staticmethod
    def _rank_question(x_row, M, w):
        for i, user in enumerate(M[:, 0]):
            M[i, 1:-1] = Engine._compute_feature_row_for_user(user, x_row)

        # now transform to fix units (excluding users column, and the score column)
        scaler = MinMaxScaler()
        M[:, 1:-1] = scaler.fit_transform(M[:, 1:-1])

        # weight each feature
        M[:, 1: -1] = np.multiply(M[:, 1:-1], w)

        # compute the final score based off the features by summing
        # all the rows (excluding the user column and the score column)
        M[:, -1] = np.array([np.sum(M[i, 1:-1]) for i in range(M.shape[0])])

        # by default sort will sort by the last column
        return Engine._sort_matrix_by_column(M, -1, ascending=False)

    @staticmethod
    def _compute_feature_row_for_user(user_id, x_row):
        # todo: add user expertise
        user_avail = Engine.user_availability.get_user_availability_probability(user_id, x_row.CreationDate.hour)
        # user_expertise = Engine.user_expertise.get_user_sum_expertise(user_id, x_row['Tags'].split())
        user_basic_profile = Engine.user_profile.get_all_measureable_features(user_id)
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
        logging.info(f'Initial random weights: {weights}')

        t1 = time.time()
        engine = Engine()
        engine.rank_all_questions(weights, log_disabled=True)
        prev_error = engine.all_residuals.get_total_error()

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
                c_error = engine.all_residuals.get_total_error()
            return p_weight

        # first try increasing the weight
        prev_weight = w[i]
        alpha *= exp_increase
        adjust_weight(True)

        engine = Engine()
        engine.rank_all_questions(w, log_disabled=True)
        current_error = engine.all_residuals.get_total_error()
        print(current_error, prev_error)
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
            current_error = engine.all_residuals.get_total_error()
            return loop_until_error_increase(current_error, prev_error, alpha, False)


class Test:
    def __init__(self, n_features):
        self.n_features = n_features

    def build_error_matrix_by_cartisian_weight(self):
        pass

    def manual_weights(self, W):
        for w in W:
            self.manual_weight(w)

    def manual_weight(self, w):
        engine = Engine()
        engine.rank_all_questions(w)

    def random_weight(self):
        engine = Engine()
        weights = np.random.rand(1, self.n_features)[0] + 1.5
        engine.rank_all_questions(weights)

    def random_weight_with_visual_residual(self):
        engine = Engine()
        weights = np.random.rand(1, self.n_features)[0] + 1.5
        engine.rank_all_questions(weights)
        engine.build_and_display_residual_matrix('TEST_random_weight_with_visual_residual.png')

    def weight_tune(self):
        engine = Engine()
        w = WeightVector.tune_weight_vector(self.n_features)
        engine.rank_all_questions(w)

    def weight_tune_with_visual_residual(self):
        engine = Engine()
        w = WeightVector.tune_weight_vector(self.n_features)
        engine.rank_all_questions(w)
        engine.build_and_display_residual_matrix('TEST_weight_tune_with_visual_residual.png')


def primative_tests(n_features):
    my_test = Test(n_features)
    # my_test.manual_weight(np.array([30, 100, 0, 20, 1000]))
    my_test.random_weight()
    # my_test.random_weight_with_visual_residual()
    # my_test.weight_tune()
    # my_test.weight_tune_with_visual_residual()


def set_up_log_file(name):
    # delete old log file, if existing
    try:
        os.remove('engine_log.log')
    except OSError:
        pass

    # remove all handlers associated with the root logger object
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=name, level=logging.INFO)


'''
Feature Summary:
  BasicProfile (4):
    - reputation
    - views
    - up votes
    - down votes
  UserExpertise (1) ignored for now
  UserAvailability (1)
'''

if __name__ == '__main__':
    set_up_log_file('engine_log.log')
    primative_tests(5)
