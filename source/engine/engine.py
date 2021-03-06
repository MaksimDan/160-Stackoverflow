# data structures
import time
import json
from features import *
from visuals import DataUtilities

# utilities
from collections import defaultdict
from copy import copy
import os
import itertools
import math

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
        self.full_raw_residuals_per_question = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: list())))

    def compute_and_store_residuals_filtered(self, score_matrix, y_index):
        """
        objective: store limited information residuals for plotting purposes
        :param score_matrix: NxM np matrix
        :param y_index: int - question number
        :return:
        """
        observed = self.y.iloc[y_index]
        predicted = {int(score_matrix[i, 0]): {'index': i, 'score': score_matrix[i, -1]}
                     for i in range(score_matrix.shape[0])}

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
        # self.error_per_question['favorite'].append(sum(index_favorite))

        self.raw_residuals_per_question[y_index]['i_answer'] = index_answer
        self.raw_residuals_per_question[y_index]['i_comment'] = index_comment
        self.raw_residuals_per_question[y_index]['i_editor'] = index_editor
        # self.raw_residuals_per_question[y_index]['i_favorite'] = index_favorite

    def compute_and_store_residuals_unfiltered(self, score_matrix, y_index):
        """
        objective: store residual information related to residual analysis processing
        :param score_matrix: NxM np matrix
        :param y_index: int - question number
        :return:
        """
        observed = self.y.iloc[y_index]
        predicted = {int(score_matrix[i, 0]): {'index': i, 'score': score_matrix[i, -1]}
                     for i in range(score_matrix.shape[0])}

        def try_iter(l):
            indices, users, scores = [], [], []
            for user in l:
                try:
                    indices.append(predicted[user]['index'])
                    users.append(user)
                    scores.append(predicted[user]['score'])
                except KeyError:
                    continue
            return indices, users, scores

        i_ans, u_ans, s_ans = try_iter(eval(observed.answerers))
        i_com, u_com, s_com = try_iter(eval(observed.commenters))
        i_ed, u_ed, s_ed = try_iter(eval(observed.editers))
        # i_fav, u_fav, s_fav = try_iter(eval(observed.favorite))

        self.full_raw_residuals_per_question[y_index]['answer']['rank'] = i_ans
        self.full_raw_residuals_per_question[y_index]['answer']['userid'] = u_ans
        self.full_raw_residuals_per_question[y_index]['answer']['score'] = s_ans

        self.full_raw_residuals_per_question[y_index]['comment']['rank'] = i_com
        self.full_raw_residuals_per_question[y_index]['comment']['userid'] = u_com
        self.full_raw_residuals_per_question[y_index]['comment']['score'] = s_com

        self.full_raw_residuals_per_question[y_index]['edit']['rank'] = i_ed
        self.full_raw_residuals_per_question[y_index]['edit']['userid'] = u_ed
        self.full_raw_residuals_per_question[y_index]['edit']['score'] = s_ed

        # removing favorites as part of the error (doesnt make much sense to route)
        # self.full_raw_residuals_per_question[y_index]['favorite']['rank'] = i_fav
        # self.full_raw_residuals_per_question[y_index]['favorite']['userid'] = u_fav
        # self.full_raw_residuals_per_question[y_index]['favorite']['score'] = s_fav

    def get_loss_function_errors(self, t_discrete):
        """
        objective: return all the loss function errors per question
        :param t_discrete: int - population threshold criteria
        :return: list - loss function error per function
        """
        def flatten_lists(lists):
            return list(itertools.chain.from_iterable(lists))

        errors = []
        for y_index, activities in self.raw_residuals_per_question.items():
            obs_q = flatten_lists(activities.values())
            if len(obs_q) != 0:
                errors.append(1 - (sum([1 if 0 <= obs <= t_discrete else 0 for obs in obs_q]) / len(obs_q)))
        return errors

    def get_loss_function_total_error(self, t):
        """
        objective: aggregates together all loss function errors
        :param t: int - population threshold
        :return: float - total loss function error
        """
        return sum(self.get_loss_function_errors(t))

    def get_total_rank_error(self):
        """
        objective: aggregates together all ranks
        :return: float - total sum of ranks
        """
        d = self.flatted_errors()
        return sum([d[key] for key in d.keys()])

    def get_total_observed_users(self):
        """
        objective: counts the total number of user activities
        :return: int - total activities
        """
        return sum([len(indices) for y_index, activities in self.raw_residuals_per_question.items()
                    for activity, indices in activities.items()])

    def get_summarize_statistics(self, n_total_users):
        """
        objective: print performance summary for log file
        :param n_total_users: int - population length
        :return:
        """
        n_questions = len(self.raw_residuals_per_question)

        stats = 'Residual Summary Statistics\n\n'

        stats += f'----- META -----\n'
        stats += f'Total Questions Analyzed: {n_questions}\n'
        stats += f'Total Users Ranked: {n_total_users}\n\n'

        stats += '----- RANK -----\n'
        stats += f'Summation of Ranks: {self.get_total_rank_error()}\n'

        stats += 'Error per User Activity:\n'
        stats += Residuals.pprint_dict(self.flatted_errors())

        stats += '\nAverage Percent Error per User Activity:\n'
        stats += Residuals.pprint_dict(self.get_average_percent_rank_error_per_question(n_total_users))

        stats += '\nThreshold Accuracy per User Activity:\n'
        stats += Residuals.pprint_dict(self.summarize_error_by_threshold())

        stats += f'\nAverage Error Per Question: {self.get_total_rank_error() / n_questions}\n\n'

        stats += '----- LOSS FUNCTION -----\n'
        threshold_rank = math.floor(n_total_users*.17)
        loss_errors = self.get_loss_function_errors(threshold_rank)
        loss_error = sum(loss_errors)

        stats += f'Threshold Choice: {threshold_rank}\n'
        stats += f'Total Error: {loss_error}\n'
        stats += f'Average ratio of missed users per question: {np.mean(loss_errors)}\n'
        stats += f'Standard Deviation of missed users per question: {np.std(loss_errors)}\n\n'

        total_user_activities = self.get_total_observed_users()
        hear_back_from = math.floor(total_user_activities * (1 - np.mean(loss_errors)))
        stats += f'If we send out an email to {threshold_rank} out of {n_total_users} total users,\n' + \
                 f'then we can expect to meet {(1 - np.mean(loss_errors)) * 100:.4f}% of our target responders.\n' + \
                 f'From the identified {total_user_activities} total user activities, we can expect to hear back\n' + \
                 f'from {hear_back_from} such users.'
        return stats

    def flatted_errors(self):
        """
        objective: aggregate ranks per activity basis
        :return: dict - sum of ranks by activity
        """
        flatted_errors = defaultdict(int)
        for activity, ranks in self.error_per_question.items():
            flatted_errors[activity] = sum(ranks)
        return flatted_errors

    def get_average_percent_rank_error_per_question(self, n_total_users):
        """
        objective: compute the average rank error per question
        :param n_total_users: int - total users
        :return:
        """
        d = {}
        for activity, ranks in self.error_per_question.items():
            d[activity] = np.mean(ranks) / n_total_users
        return d

    def summarize_error_by_threshold(self):
        """
        objective: print information related to error by threshold
                   note: there is a plot equivalent to this
        :return: dict - summary of threshold
        """
        t_df = DataUtilities.build_threshold_dataframe(self.raw_residuals_per_question)

        threshold_summary = {}
        summary_marker = np.arange(0, 1 + .1, .1)
        for activity, sub_df in t_df.groupby('activity'):
            # normalize the positions where the thresholds are, in order to observe relative position
            _max_position = max(sub_df['capture_accuracy'].values)
            sub_df['capture_accuracy'] = sub_df['capture_accuracy'].apply(lambda x: x/_max_position)
            basic_increment_index = np.where(np.isin(np.array(sub_df['t']), summary_marker))
            summary_threshold = np.array(sub_df['capture_accuracy'])[basic_increment_index]
            threshold_summary[activity] = {f'{t*100:.4f}%': t_error
                                           for t, t_error in zip(summary_marker, summary_threshold)}
        return threshold_summary

    def build_label_matrix(self, matrix_init):
        """
        objective: build NxM matrix that identifies whether there or was
                   an activity at question_i, user_j
        :param matrix_init: NxM np zero matrix
        :return:
        """
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
    X_full = pd.read_csv(BASE_PATH + 'X_train.csv')
    y_full = pd.read_csv(BASE_PATH + 'y_train.csv')
    X = X_full.head(800)
    y = y_full.head(800)
    X['CreationDate'] = pd.to_datetime(X['CreationDate'], format="%Y-%m-%dT%H:%M:%S")

    # load engineered features
    user_availability = UserAvailability()
    user_profile = BasicProfile()
    user_expertise = UserExpertise()
    user_q_relation = UserQuestionRelation()

    # loading post features
    indicator = Indicator()

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
        if visuals_active or save_feature_matrices:
            self.recommender_user_matrix = np.zeros((len(Engine.X), len(Engine.unique_users_list)), dtype=np.int)
            self.recommender_score_matrix = np.zeros((len(Engine.X), len(Engine.unique_users_list)), dtype=np.float)
            self.recommender_label_matrix = np.zeros((len(Engine.X), len(Engine.unique_users_list)), dtype=np.int)

    def rank_all_questions(self, w):
        """
        objective: rank all the users on per question basis
        :param w: np.array - weight vector for features
        :return:
        """
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
            question_score_matrix = Engine._rank_question(index, row, copy(matrix_init), w)
            if not self.save_feature_matrices:
                self.residuals.compute_and_store_residuals_filtered(question_score_matrix, index)
            else:
                self.residuals.compute_and_store_residuals_unfiltered(question_score_matrix, index)
                np.savetxt(f'./feature_matrices/q_{index}_feature_matrix.csv', question_score_matrix, delimiter=',')

            # store the user matrix and score results for entropy visualization and roc curve
            if self.visuals_active or self.save_feature_matrices:
                self.recommender_user_matrix[index, :] = question_score_matrix[:, 0]
                self.recommender_score_matrix[index, :] = question_score_matrix[:, -1]
        t2 = time.time()

        # finally add labels for residuals for roc curve
        if self.visuals_active or self.save_feature_matrices:
            self.residuals.build_label_matrix(self.recommender_label_matrix)

        logging.info(f'Ranking all questions computation finished in {(t2 - t1)/60} minutes.\n'
                     f'With an average time of {(t2 - t1)/len(Engine.X)} seconds per question.')
        if not self.log_disabled:
            logging.info(self.residuals.get_summarize_statistics(len(Engine.unique_users_list)))
        logger.disabled = False

    @staticmethod
    def _rank_question(q_num, x_row, M, w):
        """
        objective: rank individual question
        :param q_num: int - question index or number
        :param x_row: pd.Series - feature row for question
        :param M: NxM feature matrix
        :param w: np.array - weight vector
        :return: NxM feature matrix that consists of users and feature scores
        """
        for i, user in enumerate(M[:, 0]):
            M[i, 1:-1] = Engine._compute_feature_row_for_user(q_num, user, x_row)

        # now transform to fix units (excluding users column, and the score column)
        scaler = MinMaxScaler()
        M[:, 1: -1] = scaler.fit_transform(M[:, 1:-1])

        # weight each feature (except the user and score)
        M[:, 1: -1] = np.multiply(M[:, 1:-1], w)

        # compute the final score based off the features by summing
        # all the rows (excluding the user column and the score column)
        M[:, -1] = np.array([np.sum(M[i, 1:-1]) for i in range(M.shape[0])])

        # post process the matrix, and reduce inactive users before sorting the scores
        M[:, -1] = Engine.__post_process_indicate(M, x_row)

        # by default sort will sort by the last column
        return Engine._sort_matrix_by_column(M, -1, ascending=False)

    @staticmethod
    def _compute_feature_row_for_user(q_num, user_id, x_row):
        """
        objective: compute the features scores for a user
        :param q_num: int - question number or index
        :param user_id: int - user identifier
        :param x_row: pd.Series - feature meta information
        :return:
        """
        user_avail = Engine.user_availability.get_user_availability_probability(user_id, x_row.CreationDate.hour)
        # deprecation note: no longer using user basic profile as a feature (worthless)
        # user_basic_profile = Engine.user_profile.get_all_measureable_features(user_id)
        user_expertise_a = Engine.user_expertise.get_user_sum_expertise(user_id, x_row['Tags'].split(), 'n_answers')
        user_expertise_c = Engine.user_expertise.get_user_sum_expertise(user_id, x_row['Tags'].split(), 'n_comments')
        user_expertise_q = Engine.user_expertise.get_user_sum_expertise(user_id, x_row['Tags'].split(), 'n_questions')

        user_sim_expertise_a = Engine.user_expertise.get_user_sum_tag_sim_expertise(user_id, x_row['Tags'].split(), 'n_answers')
        user_sim_expertise_c = Engine.user_expertise.get_user_sum_tag_sim_expertise(user_id, x_row['Tags'].split(), 'n_comments')
        user_sim_expertise_q = Engine.user_expertise.get_user_sum_tag_sim_expertise(user_id, x_row['Tags'].split(), 'n_questions')

        user_a_relation = Engine.user_q_relation.get_user_question_relation(q_num, user_id, 'answers')
        user_q_relation = Engine.user_q_relation.get_user_question_relation(q_num, user_id, 'questions')
        user_t_relation = Engine.user_q_relation.get_user_question_relation(q_num, user_id, 'titles')
        user_c_relation = Engine.user_q_relation.get_user_question_relation(q_num, user_id, 'comments')

        return [user_avail, user_expertise_a, user_expertise_c, user_expertise_q,
                user_sim_expertise_a, user_sim_expertise_c, user_sim_expertise_q,
                user_a_relation, user_q_relation, user_t_relation, user_c_relation]

    @staticmethod
    def _sort_matrix_by_column(M, i_column, ascending=True):
        """
        objective: sort a matrix by a column name
        :param M: NxM matrix
        :param i_column: int - index column to sort by
        :param ascending: boolean - true if ascending order
        :return:
        """
        multiplier = 1 if ascending else -1
        return M[np.argsort(multiplier*M[:, i_column])]

    @staticmethod
    def __post_process_indicate(M, x_row):
        """
        objective: filter through feature matrix by indicator
        :param M: NxM feature matrix
        :param x_row: pd.Series - feature meta information
        :return: np.array - new filtered out scores
        """
        # key:
        #   M[i, 0] -> user_id
        #   M[i, -1] -> score
        return np.array([0 if Engine.indicator.is_inactive(x_row.Id, M[i, 0]) else M[i, -1]
                         for i in range(M.shape[0])])
