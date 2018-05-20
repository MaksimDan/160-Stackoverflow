import dill as pickle
import sys

# preprocessing in order for the pickle files to unpack
# the modules must be included where they were packed
sys.path.append('../source/data/features')
BASE_PATH = '../../160-Stackoverflow-Data/train_test/'


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
            self.UA_network[int(user_id)]
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