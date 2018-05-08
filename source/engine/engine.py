import pandas as pd
import seaborn as sns
import json

class ResidualMatrix:
    pass


class ResidualStrip:
    pass


class Feature:
    pass


class UserAvailability(Feature):
    JSON_PATH = '../../160-Stackoverflow-Data/train_test/engineered_features/user_availability_network.csv'

    def __init__(self):
        with open(UserAvailability.JSON_PATH) as f:
            UA_network = json.load(f)


class UserExpertise(Feature):
    JSON_PATH = '../../160-Stackoverflow-Data/train_test/engineered_features/user_expertise_network.csv'

    def __init__(self):
        with open(UserExpertise.JSON_PATH) as f:
            UE_network = json.load(f)

    def get_user_expertise(self, user_id, tag):
        pass


class Reputation(Feature):
    USERS_PATH = '../../160-Stackoverflow-Data/train_test/engineered_features/user_expertise_network.csv'

    def __init__(self):
        with open(UserExpertise.JSON_PATH) as f:
            self.UE_network = json.load(f)

    def get_user_reputation(self, userid):
        pass


class Engine:
    def __init__(self):
        self.X = pd.read_csv('../../160-Stackoverflow-Data/train_test/X.csv')
        self.y = pd.read_csv('../../160-Stackoverflow-Data/train_test/y.csv')

    def build_score_dataframe(self):
        pass

    def build_residual_matrix(self):
        pass
