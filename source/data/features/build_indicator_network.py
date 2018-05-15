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

BASE_PATH = '../../160-Stackoverflow-Data/train_test/'


class I:
    # the data
    Users = pd.read_csv(BASE_PATH + 'Users.csv')
    Posts = pd.read_csv(BASE_PATH + 'Posts_Clean.csv')
    all_users = pickle.load(open(BASE_PATH + 'meta/users_list.p', 'rb'))

    # date preprocessing
    Posts.CreationDate = pd.to_datetime(Posts.CreationDate, format="%Y-%m-%dT%H:%M:%S")
    Users.CreationDate = pd.to_datetime(Users.LastAccessDate, format="%Y-%m-%dT%H:%M:%S")

    # date subsetting
    Questions = Posts.loc[Posts.PostTypeId == 1]
    Answers = Posts.loc[Posts.PostTypeId == 2]

    @staticmethod
    # this feature should
    def is_inactive_user(user_id, question_id):
        try:
            question_creation_date = I.Questions.loc[I.Questions.Id == question_id].CreationDate.values[0]
            user_creation_date = I.Users.loc[I.Users.Id == user_id].CreationDate.values[0]
            user_last_access_date = I.Users.loc[I.Users.Id == user_id].LastAccessDate.values[0]

            user_create_account_before_question = user_creation_date < question_creation_date
            user_access_account_before_question = user_creation_date < user_last_access_date

            return int(user_create_account_before_question and user_access_account_before_question)
        except:
            return None


def build_indicator_network():
    inactive_by_post_id = defaultdict(set)
    all_questions, all_users = I.Questions, I.all_users

    bar = progressbar.ProgressBar()
    for question_id in bar(all_questions.Id.values):
        for user_id in all_users:
            if I.is_inactive_user(user_id, question_id):
                inactive_by_post_id[question_id].add(user_id)

    with open('indicator_network.p', 'wb') as fp:
        pickle.dump(inactive_by_post_id, fp)


build_indicator_network()
