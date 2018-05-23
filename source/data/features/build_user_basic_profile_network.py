import pandas as pd
import progressbar
from collections import defaultdict
import dill as pickle


"""
File: build_user_basic_profile_network.py
Objective: Build a json file that identifies the basic
           meta profile properties associated with a user.

Graph Structure:
    {
        <user_id1> = {'prop1': val1, 'prop2': val2, ... },
        <user_id1> = {'prop1': val1, 'prop2': val2, ... },
        ...
        <user_idn> = {'prop1': val1, 'prop2': val2, ... },
    }
"""


def build_user_basic_profile_network():
    # load data
    from build_all_features import BASE_PATH
    Users = pd.read_csv(BASE_PATH + 'raw_query/Users.csv')

    # only interested in the user and the creation dates
    user_basic_profile = defaultdict(lambda: defaultdict(lambda: int()))

    bar = progressbar.ProgressBar()
    for index, row in bar(Users.iterrows()):
        user_basic_profile[int(row['Id'])]['reputation'] = row.Reputation
        user_basic_profile[int(row['Id'])]['views'] = row.Views
        user_basic_profile[int(row['Id'])]['creation_date'] = row.CreationDate
        user_basic_profile[int(row['Id'])]['upvotes'] = row.UpVotes
        user_basic_profile[int(row['Id'])]['downvotes'] = row.DownVotes

    with open('user_basic_profile_network.p', 'wb') as fp:
        pickle.dump(user_basic_profile, fp)
