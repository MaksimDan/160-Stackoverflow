import pandas as pd
import dill as pickle
from collections import defaultdict
import progressbar


"""
File: build_indicator_network.py
Objective: Build a dictionary file that identifies users who  
           cannot possibly have any associated level of activity
           connecting to a question.
Graph Structure:
    {
        <question_id1> = {user1, user2, ..., usern},
        <question_id2> = {user1, user2, ..., usern},
        ...
        <question_id3> = {user1, user2, ..., usern},
    }
"""


def go(postid_date, userid_date, Questions, BASE_PATH):
    def user_created_account_after_question(user_id, question_id):
        try:
            return postid_date[question_id] < userid_date[user_id]['CreationDate']
        except KeyError:
            return False

    def user_inactive_before_question(user_id, question_id):
        try:
            return userid_date[user_id]['LastAccessDate'] < postid_date[question_id]
        except KeyError:
            return False

    i_dict = defaultdict(set)
    all_questions, all_users = Questions.Id.values, pickle.load(open(BASE_PATH + 'meta/users_list.p', 'rb'))

    bar = progressbar.ProgressBar()
    for question_id in bar(all_questions):
        for user_id in all_users:
            if user_created_account_after_question(user_id, question_id): #or \
                    #user_inactive_before_question(user_id, question_id):
                i_dict[question_id].add(user_id)

    with open('indicator_network.p', 'wb') as fp:
        pickle.dump(i_dict, fp)


def build_indicator_network():
    # the data
    from build_all_features import BASE_PATH
    Users = pd.read_csv(BASE_PATH + 'raw_query/Users.csv')
    Questions = pd.read_csv(BASE_PATH + 'X_train.csv').head(2000)

    # date preprocessing
    Questions.CreationDate = pd.to_datetime(Questions.CreationDate, format="%Y-%m-%dT%H:%M:%S")
    Users.CreationDate = pd.to_datetime(Users.CreationDate, format="%Y-%m-%dT%H:%M:%S")
    Users.LastAccessDate = pd.to_datetime(Users.LastAccessDate, format="%Y-%m-%dT%H:%M:%S")

    # preproprocessing lookups to speed things up
    postid_date = {row['Id']: row['CreationDate'] for index, row in Questions.iterrows()}
    userid_date = {row['Id']: {'CreationDate': row['CreationDate'],
                               'LastAccessDate': row['LastAccessDate']} for index, row in Users.iterrows()}

    del Users
    go(postid_date, userid_date, Questions, BASE_PATH)
