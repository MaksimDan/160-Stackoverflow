import pandas as pd
import json
import sys
from collections import defaultdict


class ResidualMatrix:
    pass


class ResidualStrip:
    pass


class Feature:
    pass


class UserAvailability(Feature):
    JSON_PATH = '../../160-Stackoverflow-Data/train_test/engineered_features/user_availability_network.csv'

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
    JSON_PATH = '../../160-Stackoverflow-Data/train_test/engineered_features/user_expertise_static_network.csv'

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
    USERS_PATH = '../../160-Stackoverflow-Data/train_test/Users.csv'

    def __init__(self):
        try:
            with open(BasicProfile.USERS_PATH) as f:
                self.Users = json.load(f)
        except OSError as e:
            print(f'{BasicProfile.USERS_PATH} was not found.', e)
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
        self.X = pd.read_csv('../../160-Stackoverflow-Data/train_test/X.csv')
        self.y = pd.read_csv('../../160-Stackoverflow-Data/train_test/y.csv')

    # todo: use timeit do see where things are running inefficiently

    def build_user_score_matrix(self, weight_vector):
        # todo: run standard scaler here
        pass

    def build_residual_matrix(self):
        pass
