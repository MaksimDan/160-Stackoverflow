import json
import time

# meta
import logging
import progressbar
import pickle

# data
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class ResidualMatrix:
    pass


class ResidualStrip:
    pass


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
            return self.UA_network[str(user_id)][str(hour)]
        except KeyError as e:
            print(f'Either user_id {user_id} or hour {str(hour)} was not found.', e)
            raise


class UserExpertise(Feature):
    JSON_PATH = '../../160-Stackoverflow-Data/train_test/engineered_features/user_expertise_network.json'

    def __init__(self):
        try:
            with open(UserExpertise.JSON_PATH) as f:
                self.UE_network = json.load(f)
        except OSError as e:
            print('Warning: User Expertise file was not found.', e)

    def get_user_expertise(self, user_id, tag):
        try:
            return self.UE_network[str(user_id)][tag]
        except KeyError as e:
            print(f'Either user_id {user_id} or hour {tag} was not found.', e)
            raise


class BasicProfile(Feature):
    JSON_PATH = '../../160-Stackoverflow-Data/train_test/user_basic_profile_network.json'

    def __init__(self):
        try:
            with open(BasicProfile.JSON_PATH) as f:
                self.Users = json.load(f)
        except OSError as e:
            print(f'{BasicProfile.JSON_PATH} was not found.', e)
            raise

    def get_user_reputation(self, user_id):
        try:
            return self.Users[str(user_id)]['reputation']
        except KeyError as e:
            print(f'user_id {user_id} was not found.', e)
            raise

    def get_creation_date(self, user_id):
        try:
            return self.Users[str(user_id)]['creation_date']
        except KeyError as e:
            print(f'user_id {user_id} was not found.', e)
            raise

    def get_views(self, user_id):
        try:
            return self.Users[str(user_id)]['views']
        except KeyError as e:
            print(f'user_id {user_id} was not found.', e)
            raise

    def get_upvotes(self, user_id):
        try:
            return self.Users[str(user_id)]['upvotes']
        except KeyError as e:
            print(f'user_id {user_id} was not found.', e)
            raise

    def get_downvotes(self, user_id):
        try:
            return self.Users[str(user_id)]['downvotes']
        except KeyError as e:
            print(f'user_id {user_id} was not found.', e)
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


class Engine:
    def __init__(self):
        logging.info('Loading all dataframes...')
        t1 = time.time()
        # load questions and all user activities
        self.X = pd.read_csv('../../160-Stackoverflow-Data/train_test/X.csv')
        self.y = pd.read_csv('../../160-Stackoverflow-Data/train_test/y.csv')

        # load engineered features
        self.user_availability = UserAvailability()
        self.user_expertise = UserExpertise()
        self.user_profile = BasicProfile()
        self.tag_network = TagNetwork()
        t2 = time.time()
        logging.info(f'Dataframe loading finished in time {t2 - t1} seconds.')

    def rank(self, n_features, weight_vector):
        logging.info('Computing ranks...')
        t1 = time.time()

        n_questions = len(self.X)
        bar = progressbar.ProgressBar()
        for index, row in bar(self.X.iterrows()):
            # n_features + 2, for the user id column and score oolumn
            matrix_init = np.zeros(shape=(n_questions, n_features + 2), dtype=np.float)

            # add all the 
            with open('users_list.p', 'rb') as fp:
                matrix_init[:, 0] = pickle.load(fp)
            user_score_matrix = self._build_user_score_matrix(row, matrix_init, weight_vector)

        t2 = time.time()
        logging.info(f'Ranking computation finished in {(t2 - t1)/60} minutes.')


    def _build_user_score_matrix(self, question, M, weight_vector):
        for i, user in enumerate(M[:, 0]):
            M[i, 1:-1] = self.__compute_feature_row_for_user(user)

        # now transform to fix units (excluding users column, and the score column)
        scaler = StandardScaler()
        scaler.fit_transform(M[:, 1:-1])

        # weight each feature
        M[:, 1: -1] = M[:, 1:-1]*weight_vector

        # compute the final score based off the features by summing
        # all the rows (excluding the user column and the score column)
        M[:, -1] = [sum(np.sum(M[i, 1:-1])) for i in range(M.shape[0])]

        # by default sort will sort by the last column
        return self._sort_matrix_by_column(M, -1, ascending=False)

    def __compute_feature_row_for_user(self, user_id):
        pass

    def _build_residual_matrix(self):
        pass

    def _sort_matrix_by_column(self, M, i_column, ascending=True):
        multiplier = 1 if ascending else -1
        return M[np.argsort(multiplier*M.A[:, i_column])]

    def _compute_residuals(self, rank, observed):
        pass


if __name__ == '__main__':
    logging.basicConfig(filename='engine.log', level=logging.INFO)
    engine = Engine()
