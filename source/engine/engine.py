# data structures
import time
import json
import pickle

from features import BASE_PATH, UserAvailability, UserExpertise, BasicProfile
from visuals import ResidualPlots
from residuals import Residuals

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




class Engine:
    logging.info('Loading all data frames...')
    t1 = time.time()

    # load questions and all user activities
    X = pd.read_csv(BASE_PATH + 'X_train.csv').head(18)
    y = pd.read_csv(BASE_PATH + 'y_train.csv').head(18)
    X['CreationDate'] = pd.to_datetime(X['CreationDate'], format="%Y-%m-%dT%H:%M:%S")

    # load engineered features
    user_availability = UserAvailability()
    user_profile = BasicProfile()
    user_expertise = UserExpertise()

    # todo: ignore these for now, integrate them later once everything works
    # tag_network = TagNetwork()

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
        for index, row in bar(Engine.X.iterrows()):
            question_score_matrix = Engine.rank_question(row, copy(matrix_init), w)
            self.residuals.compute_and_store_residuals(question_score_matrix, index)

        t2 = time.time()
        logging.info(f'Ranking all questions computation finished in {(t2 - t1)/60} minutes.\n'
                     f'With an average time of {(t2 - t1)/len(Engine.X)} seconds per question.')
        if not log_disabled:
            logging.info(self.residuals.get_summarize_statistics(len(Engine.unique_users_list)))
        logger.disabled = False
        with open('user_variabity.json', 'w+') as file:
            json.dump(self.residuals.user_rank, file)

    @staticmethod
    def rank_question(x_row, M, w):
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
        user_expertise = Engine.user_expertise.get_user_sum_expertise(user_id, x_row['Tags'].split())
        user_basic_profile = Engine.user_profile.get_all_measureable_features(user_id)
        return [user_avail] + user_basic_profile + [user_expertise]

    @staticmethod
    def _sort_matrix_by_column(M, i_column, ascending=True):
        multiplier = 1 if ascending else -1
        return M[np.argsort(multiplier*M[:, i_column])]
