# data structures
import time
import json
from features import *
from visuals import ResidualPlots

# utilities
from collections import defaultdict
from copy import copy
import os

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
        observed = self.y.iloc[y_index]
        predicted = {int(score_matrix[i, 0]): {'index': i, 'score': score_matrix[i, -1]} for i in range(score_matrix.shape[0])}

        def try_iter(l):
            indices = []
            for user in l:
                try:
                    indices.append(predicted[user]['index'])
                except KeyError:
                    continue
            return indices

        index_answer = try_iter(eval(observed.answerers))
        index_comment = try_iter(eval(observed.commenters))
        index_editor = try_iter(eval(observed.editers))
        index_favorite = try_iter(eval(observed.favorite))

        self.error_per_question['answer'].append(sum(index_answer))
        self.error_per_question['comment'].append(sum(index_comment))
        self.error_per_question['edit'].append(sum(index_editor))
        self.error_per_question['favorite'].append(sum(index_favorite))

        self.raw_residuals_per_question[y_index]['i_answer'] = index_answer
        self.raw_residuals_per_question[y_index]['i_comment'] = index_comment
        self.raw_residuals_per_question[y_index]['i_editor'] = index_editor
        self.raw_residuals_per_question[y_index]['i_favorite'] = index_favorite

    def get_total_error(self):
        d = self.flatted_errors()
        return sum([d[key] for key in d.keys()])

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
        d = {}
        for activity, ranks in self.error_per_question.items():
            d[activity] = np.mean(ranks) / n_total_users
        return d

    def summarize_error_by_threshold(self):
        r_plot = ResidualPlots()
        t_df = r_plot.build_threshold_dataframe(self.raw_residuals_per_question)

        threshold_summary = {}
        summary_marker = np.arange(0, 1 + .1, .1)
        for activity, sub_df in t_df.groupby('activity'):
            # normalize the positions where the thresholds are, in order to observe relative position
            _max_position = max(sub_df['capture_accuracy'].values)
            sub_df['capture_accuracy'] = sub_df['capture_accuracy'].apply(lambda x: x/_max_position)
            basic_increment_index = np.where(np.isin(np.array(sub_df['t']), summary_marker))
            summary_threshold = np.array(sub_df['capture_accuracy'])[basic_increment_index]
            threshold_summary[activity] = {f'{round(t*100, 1)}%': t_error
                                           for t, t_error in zip(summary_marker, summary_threshold)}
        return threshold_summary

    def build_label_matrix(self, matrix_init):
        # single classification for now (treat any kind of user activity as the same activity)
        for question_number, activities in self.raw_residuals_per_question.items():
            for activity, indices in activities.items():
                for index in indices:
                    matrix_init[question_number, index] = 1

    @staticmethod
    def pprint_dict(d):
        return json.dumps(d, indent=4)


class Engine:
    logging.info('Loading all data frames...')
    t1 = time.time()

    # load questions and all user activities
    X = pd.read_csv(BASE_PATH + 'X_train.csv').head(35) #.sample(35, random_state=42)
    y = pd.read_csv(BASE_PATH + 'y_train.csv').head(35) #.sample(35, random_state=42)
    X['CreationDate'] = pd.to_datetime(X['CreationDate'], format="%Y-%m-%dT%H:%M:%S")

    # load engineered features
    user_availability = UserAvailability()
    user_profile = BasicProfile()
    user_expertise = UserExpertise()

    # loading post features
    indicator = Indicator()

    # todo: ignore these for now, integrate them later once everything works
    # tag_network = TagNetwork()

    # load all users list (order does not matter)
    with open(BASE_PATH + 'meta/users_list.p', 'rb') as fp:
        unique_users_list = pickle.load(fp)

    t2 = time.time()
    logging.info(f'DataFrame loading finished in time {t2 - t1} seconds.\n')

    def __init__(self, log_disabled=False, save_feature_matrices=False,
                 visuals_active=True):
        self.log_disabled, self.save_feature_matrices, \
            self.visuals_active = log_disabled, save_feature_matrices, visuals_active

        self.residuals = Residuals(Engine.X, Engine.y)
        if visuals_active:
            self.recommender_user_matrix = np.zeros((len(Engine.X), len(Engine.unique_users_list)))
            self.recommender_score_matrix = np.zeros((len(Engine.X), len(Engine.unique_users_list)))
            self.recommender_label_matrix = np.zeros((len(Engine.X), len(Engine.unique_users_list)))

    def rank_all_questions(self, w, opt_activity):
        if self.save_feature_matrices and not os.path.exists('feature_matrices'):
            os.makedirs('feature_matrices')

        logger = logging.getLogger()
        logger.disabled = self.log_disabled
        n_features = len(w)

        # n_features + 2, for the user id column and score column
        matrix_init = np.zeros(shape=(len(Engine.unique_users_list), n_features + 2), dtype=np.float)
        # add the users list as the first column of matrix
        matrix_init[:, 0] = np.array(Engine.unique_users_list)

        logging.info(f'Computing ranks using weight vector: {w}')

        # iterate through all questions
        t1 = time.time()
        bar = progressbar.ProgressBar()
        for index, row in bar(Engine.X.iterrows()):
            question_score_matrix = Engine._rank_question(row, copy(matrix_init), w, opt_activity)
            self.residuals.compute_and_store_residuals(question_score_matrix, index)

            if self.visuals_active:
                # store the user matrix and score results for entropy visualization and roc curve
                self.recommender_user_matrix[index, :] = question_score_matrix[:, 0]
                self.recommender_score_matrix[index, :] = question_score_matrix[:, -1]

            if self.save_feature_matrices:
                np.savetxt(f'./feature_matrices/q_{index}_feature_matrix.csv', question_score_matrix, delimiter=',')
        t2 = time.time()

        if self.visuals_active:
            # finally add labels for residuals for roc curve
            self.residuals.build_label_matrix(self.recommender_label_matrix)

        logging.info(f'Ranking all questions computation finished in {(t2 - t1)/60} minutes.\n'
                     f'With an average time of {(t2 - t1)/len(Engine.X)} seconds per question.')
        logging.info(self.residuals.get_summarize_statistics(len(Engine.unique_users_list)))
        logger.disabled = False

    @staticmethod
    def _rank_question(x_row, M, w, opt_activity):
        for i, user in enumerate(M[:, 0]):
            M[i, 1:-1] = Engine._compute_feature_row_for_user(user, x_row)

        # now transform to fix units (excluding users column, and the score column)
        scaler = MinMaxScaler()
        M[:, 1: -1] = scaler.fit_transform(M[:, 1:-1])

        # weight each feature (except the user and score)
        M[:, 1: -1] = np.multiply(M[:, 1:-1], w)

        # compute the final score based off the features by summing
        # all the rows (excluding the user column and the score column)
        M[:, -1] = np.array([np.sum(M[i, 1:-1]) for i in range(M.shape[0])])

        # post process the matrix, and reduce inactive users before sorting the scores
        # M[:, -1] = Engine.__post_process_indicate(M, x_row, opt_activity)

        # by default sort will sort by the last column
        return Engine._sort_matrix_by_column(M, -1, ascending=False)

    @staticmethod
    def _compute_feature_row_for_user(user_id, x_row):
        user_avail = Engine.user_availability.get_user_availability_probability(user_id, x_row.CreationDate.hour)
        user_basic_profile = Engine.user_profile.get_all_measureable_features(user_id)
        user_expertise = Engine.user_expertise.get_user_sum_expertise(user_id, x_row['Tags'].split())
        return [user_avail] + user_basic_profile + [user_expertise]

    @staticmethod
    def _sort_matrix_by_column(M, i_column, ascending=True):
        multiplier = 1 if ascending else -1
        return M[np.argsort(multiplier*M[:, i_column])]

    @staticmethod
    def __post_process_indicate(M, x_row, opt_activity):
        # key:
        #   M[i, 0] -> user_id
        #   M[i, -1] -> score
        # after testing, we have identified that the indicator variables
        # only help reduce the error for certain kinds of activities
        if opt_activity == 'answerers' or opt_activity == 'commentors':
            return np.array([0 if (Engine.indicator.is_inactive(x_row.Id, M[i, 0]) or
                                   Engine.indicator.has_no_relative_expertise(M[i, 0], x_row.Tags.split())) else M[i, -1]
                             for i in range(M.shape[0])])
        else:
            return M[:, -1]
