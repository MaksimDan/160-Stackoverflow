# data structures
import time
import json
from features import *
from visuals import ResidualPlots

# utilities
from collections import defaultdict
from copy import copy

# meta
import logging
import progressbar

# data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


class Residuals:
    def __init__(self, X, y):
        self.X, self.y = X, y
        self.error_per_question = defaultdict(list)
        self.raw_residuals_per_question = defaultdict(lambda: defaultdict(lambda: list()))

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

        self.error_per_question['answer'].append(sum(index_answer))
        self.error_per_question['comment'].append(sum(index_comment))

        # flatten the results for ease of implementation
        self.raw_residuals_per_question[y_index]['i_answer'] = index_answer
        self.raw_residuals_per_question[y_index]['i_comment'] = index_comment

    def get_total_error(self):
        d = self.flatted_errors()
        return d['answer'] + d['comment']

    def get_summarize_statistics(self, n_total_users):
        stats = 'Residual Summary Statistics\n\n'

        stats += 'Error per User Activity:\n'
        stats += Residuals.pprint_dict(self.flatted_errors())

        stats += '\nAverage Percent Error per User Activity:\n'
        stats += Residuals.pprint_dict(self.get_average_percent_rank_error_per_question(n_total_users))

        stats += '\nThreshold Accuracy per User Activity:\n'
        stats += Residuals.pprint_dict(self.summarize_error_by_threshold())

        n_questions = len(self.raw_residuals_per_question)
        stats += '\nAverage Rank Error Per Question: '
        stats += str(self.get_total_error()/n_questions) + '\n'

        stats += 'Total Error: '
        stats += str(self.get_total_error()) + '\n'

        return stats

    def flatted_errors(self):
        flatted_errors = defaultdict(int)
        for activity, ranks in self.error_per_question.items():
            flatted_errors[activity] = sum(ranks)
        return flatted_errors

    def get_average_percent_rank_error_per_question(self, n_total_users):
        d = self.flatted_errors()
        n_questions = len(self.raw_residuals_per_question)
        for activity, sum_rank_error in d.items():
            d[activity] = sum_rank_error/n_questions/n_total_users
        return d

    def summarize_error_by_threshold(self):
        r_plot = ResidualPlots()
        t_df = r_plot.build_threshold_dataframe(self.raw_residuals_per_question)

        threshold_summary = {}
        summary_marker = np.arange(0, 1 + .1, .1)
        for activity, sub_df in t_df.groupby('activity'):
            # normalize the positions where the thresholds are, in order to observe relative position
            sub_df['position'] = np.array(sub_df['position'].values) / max(sub_df['position'].values)
            basic_increment_index = np.where(np.isin(np.array(sub_df['t']), summary_marker))
            summary_threshold = np.array(sub_df['position'])[basic_increment_index]
            threshold_summary[activity] = {f'{round(t*100, 1)}%': t_error
                                           for t, t_error in zip(summary_marker, summary_threshold)}
        return threshold_summary

    @staticmethod
    def pprint_dict(d):
        return json.dumps(d, indent=4)


class Engine:
    logging.info('Loading all data frames...')
    t1 = time.time()

    # load questions and all user activities
    X = pd.read_csv(BASE_PATH + 'X_train.csv').head(15)
    y = pd.read_csv(BASE_PATH + 'y_train.csv').head(15)
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
    logging.info(f'DataFrame loading finished in time {t2 - t1} seconds.\n')

    def __init__(self):
        self.residuals = Residuals(Engine.X, Engine.y)

    def rank_all_questions(self, w, log_disabled=False):
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

        # w_new = ','.join(str(x)[0:3] for x in w)
        for index, row in bar(Engine.X.iterrows()):
            question_score_matrix = Engine._rank_question(row, copy(matrix_init), w)
            # np.savetxt(f'r_matrix_q-{index}_w-{w_new}.csv', question_score_matrix[:, 0], delimiter=",")
            self.residuals.compute_and_store_residuals(question_score_matrix, index)

        t2 = time.time()
        logging.info(f'Ranking all questions computation finished in {(t2 - t1)/60} minutes.'
                     f'With an average time of {(t2 - t1)/len(Engine.X)} seconds per question.')
        if not log_disabled:
            logging.info(self.residuals.get_summarize_statistics(len(Engine.unique_users_list)))
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
