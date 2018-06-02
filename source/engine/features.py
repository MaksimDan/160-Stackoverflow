import dill as pickle
import sys
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp
import pandas as pd

# preprocessing in order for the pickle files to unpack
# the modules must be included where they were packed
sys.path.append('../../source/data/features')
BASE_PATH = '../../../160-Stackoverflow-Data/train_test/'


class Utilities:
    @staticmethod
    def load_p_file(p_file_path):
        try:
            with open(p_file_path, 'rb') as f:
                return pickle.load(f)
        except OSError as e:
            print(e)
            raise


class UserAvailability:
    pickle_path = BASE_PATH + 'engineered_features/user_availibility_network.p'

    def __init__(self):
        self.UA_network = Utilities.load_p_file(UserAvailability.pickle_path)

    def get_user_availability_probability(self, user_id, hour):
        try:
            self.UA_network[int(user_id)]
        except KeyError as e:
            print(f'User {e} was unidentified in the user availability network.', e)
            raise
        try:
            return self.UA_network[int(user_id)][hour]
        # if hour was not found, then there is no chance that the
        # user could have answered the question, so break the score
        except KeyError:
            return 0


class UserExpertise:
    pickle_path = BASE_PATH + 'engineered_features/user_expertise_network.p'

    def __init__(self):
        self.UE_network = Utilities.load_p_file(UserExpertise.pickle_path)
        Posts = pd.read_csv(BASE_PATH + 'raw_query/Posts.csv')
        Posts.dropna(subset=['Tags'], inplace=True)
        self.similarity_matrix, self.word2i = self._build_tag_similarity_network(Posts.Tags)
        del Posts

    def get_user_sum_expertise(self, user_id, tags, activity_type):
        return sum([self._get_user_expertise(user_id, tag, activity_type) for tag in tags])

    def get_user_sum_tag_sim_expertise(self, user_id, post_tags, activity_type):
        # to avoid any overlap with get_user_sum_expertise, this feature
        # will only be summing over tags that are different
        user_tags = self.UE_network[int(user_id)].keys()
        return sum([self._get_similarity_expertise(user_tags, post_tag, user_id, activity_type)
                    for post_tag in post_tags])

    def _get_user_expertise(self, user_id, tag, activity_type):
        try:
            user_tag_expertise = self.UE_network[int(user_id)]
            if tag not in user_tag_expertise:
                return 0
            else:
                return user_tag_expertise[tag].get(activity_type, 0.0)
                # otherwise return the sum of the frequencies of posts per tag
                # a = user_tag_expertise[tag].get('n_answers', 0.0)
                # b = user_tag_expertise[tag].get('n_comments', 0.0)
                # c = user_tag_expertise[tag].get('n_questions', 0.0)
                # return a + b
        except KeyError as e:
            print(f'user_id {user_id} was not found.', e)
            raise

    def _get_similarity_expertise(self, user_tags, post_tag, user_id, activity_type):
        return sum([self.get_tag_similarity(user_tag, post_tag) *
                    self._get_user_expertise(user_id, user_tag, activity_type)
                    if user_tag != post_tag else 0 for user_tag in user_tags])

    def get_tag_similarity(self, tag1, tag2):
        return self.similarity_matrix[self.word2i[tag1], self.word2i[tag2]]

    @staticmethod
    def _build_tag_similarity_network(docs):
        # the token pattern is necessary otherwise hyphenated tags will be ignored
        count_model = CountVectorizer(ngram_range=(1, 1), token_pattern=r'\S+')
        X = count_model.fit_transform(docs)

        # co-occurrence matrix in sparse csr format
        Xc = (X.T * X)

        # normalized co-occurence matrix
        g = sp.diags(1. / Xc.diagonal())
        Xc_norm = g * Xc

        # identify word similarity by indexing into the matrix
        word2i = {word: i for i, word in enumerate(count_model.get_feature_names())}
        return Xc_norm, word2i


class BasicProfile:
    pickle_path = BASE_PATH + 'engineered_features/user_basic_profile_network.p'
    n_features = 4

    def __init__(self):
        self.Users = Utilities.load_p_file(BasicProfile.pickle_path)

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


# post feature
class Indicator:
    pickle_path = BASE_PATH + 'engineered_features/indicator_network.p'

    def __init__(self):
        self.I_Network = Utilities.load_p_file(Indicator.pickle_path)
        self.user_avail = UserAvailability()
        self.user_expertise = UserExpertise()

    def is_inactive(self, q_id, u_id):
        # reduces error for commentors
        return u_id in self.I_Network[q_id]

    def has_no_relative_expertise(self, u_id, tags, activity_type):
        # reduces error for answerers and commentors
        return self.user_expertise.get_user_sum_expertise(u_id, tags, activity_type) == 0
