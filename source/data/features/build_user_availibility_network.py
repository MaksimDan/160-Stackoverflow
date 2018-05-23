import pandas as pd
import progressbar
import dill as pickle


"""
File: build_user_availability_network.py
Objective: Build a json file that identifies the probability
           an arbitrary user is active on hour i.

Graph Structure:
    {
        <user_id1> = {'hr1': p_hr1, 'hr2': p_hr2 ... },
        <user_id1> = {'hr1': p_hr1, 'hr2': p_hr2 ... },
        ...
        <user_idn> = {'hr1': p_hr1, 'hr2': p_hr2 ... },
    }
"""


def build_user_availibility_network():
    from build_all_features import BASE_PATH

    # load data
    Comments = pd.read_csv(BASE_PATH + 'raw_query/Comments.csv')
    Posts = pd.read_csv(BASE_PATH + 'raw_query/Posts.csv')
    Votes = pd.read_csv(BASE_PATH + 'raw_query/Votes.csv')

    # only interested in the user and the creation dates
    Full = Comments[['UserId', 'CreationDate']].append(Posts[['OwnerUserId', 'CreationDate']]).append(Votes[['UserId', 'CreationDate']])
    Full['CreationDate'] = pd.to_datetime(Full['CreationDate'], format="%Y-%m-%dT%H:%M:%S")

    user_availability = {}
    bar = progressbar.ProgressBar()
    for key, group in bar(Full.groupby('OwnerUserId')):
        activities = group['CreationDate'].dt.hour.value_counts(normalize=True)
        user_availability[int(key)] = activities.to_dict()

    with open('user_availibility_network.p', 'wb') as fp:
        pickle.dump(user_availability, fp)
