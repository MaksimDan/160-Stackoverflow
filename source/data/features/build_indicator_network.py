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

<<<<<<< HEAD
BASE_PATH = '../../160-Stackoverflow-Data/train_test/raw_query/'
=======
>>>>>>> 901709588ef2d9313e22a4333b8b5be7fa3c8b99

def build_indicator_network():
    # the data
<<<<<<< HEAD
    Users = pd.read_csv(BASE_PATH + 'Users.csv', low_memory = False)
    Posts = pd.read_csv(BASE_PATH + 'Posts.csv', low_memory = False)
    all_users = pickle.load(open(BASE_PATH + '../meta/users_list.p', 'rb'))
=======
    from build_all_features import BASE_PATH
    Users = pd.read_csv(BASE_PATH + 'raw_query/Users.csv')
    Posts = pd.read_csv(BASE_PATH + 'raw_query/Posts.csv')
    all_users = pickle.load(open(BASE_PATH + 'meta/users_list.p', 'rb'))
>>>>>>> 901709588ef2d9313e22a4333b8b5be7fa3c8b99

    # date preprocessing
    Posts.CreationDate = pd.to_datetime(Posts.CreationDate, format="%Y-%m-%dT%H:%M:%S")
    Users.CreationDate = pd.to_datetime(Users.CreationDate, format="%Y-%m-%dT%H:%M:%S")
    Users.LastAccessDate = pd.to_datetime(Users.LastAccessDate, format="%Y-%m-%dT%H:%M:%S")

    # date subsetting
    Questions = Posts.loc[Posts.PostTypeId == 1]
    Answers = Posts.loc[Posts.PostTypeId == 2]

    def user_created_account_after_question(user_id, question_id):
<<<<<<< HEAD
        question_creation_date = I.Questions.loc[I.Questions.Id == question_id].CreationDate
        user_creation_date = I.Users.loc[I.Users.Id == user_id].CreationDate
        return int(user_creation_date.iloc[0] > question_creation_date.iloc[0])
=======
        question_creation_date = Questions.loc[Questions.Id == question_id].CreationDate
        user_creation_date = Users.loc[Users.Id == user_id].CreationDate
        return int(question_creation_date < user_creation_date)
>>>>>>> 901709588ef2d9313e22a4333b8b5be7fa3c8b99

    def user_inactive_before_question(user_id, question_id):
<<<<<<< HEAD
        question_creation_date = I.Questions.loc[I.Questions.Id == question_id].CreationDate
        user_last_access_date = I.Users.loc[I.Users.Id == user_id].LastAccessDate
        return int(user_last_access_date.iloc[0] < question_creation_date.iloc[0])
=======
        question_creation_date = Questions.loc[Questions.Id == question_id].CreationDate
        user_last_access_date = Users.loc[Users.Id == user_id].LastAccessDate
        return int(user_last_access_date < question_creation_date)
>>>>>>> 901709588ef2d9313e22a4333b8b5be7fa3c8b99

    i_dict = defaultdict(lambda: defaultdict(lambda: set()))
    all_questions, all_users = Questions, all_users

    bar = progressbar.ProgressBar()
    for question_id in bar(all_questions.Id.values):
        for user_id in all_users:
            if user_created_account_after_question(user_id, question_id):
                i_dict[question_id]['create_after_q'].add(user_id)
            if user_inactive_before_question(user_id, question_id):
                i_dict[question_id]['inactive_before_q'].add(user_id)
    with open('indicator_network.p', 'wb') as fp:
        pickle.dump(i_dict, fp)

<<<<<<< HEAD
build_indicator_network()
=======
>>>>>>> 901709588ef2d9313e22a4333b8b5be7fa3c8b99
