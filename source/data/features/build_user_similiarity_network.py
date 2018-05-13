import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

import time
import progressbar
import dill as pickle


"""
File: build_user_similiarity_network.py
Objective: Identifies the n most similar users as well the associated stength
           with said user. Outputs a json graph network.

Graph Structure:
    {
        <user_id1> = {'user_id': [user1, user2, user3, ..., usern]
                      'user_weight': [weight1, weight2, weight3, ..., weightn]},
        <user_id1> = {'user_id': [user1, user2, user3, ..., usern]
                      'user_weight': [weight1, weight2, weight3, ..., weightn]},
        ...
        <user_idn> = {'user_id': [user1, user2, user3, ..., usern]
                      'user_weight': [weight1, weight2, weight3, ..., weightn]}
    }
"""


def build_n_most_similiar_users(X, df, n=15):
    user_sim = defaultdict(lambda: defaultdict(lambda: list()))

    start = time.time()
    sim_mat = (X * X.T)
    end = time.time()
    print("Similarity Matrix Finished in", end - start, "time")
    # self note: to possibly make things more efficient use
    #            np.arange(<array size>)[array[index]] to preallocate
    #            memory and then perform a vectorized index
    # I also need to convert numpy types into native python types
    # otherwise the json serializer will complain
    bar = progressbar.ProgressBar()
    for i in bar(range(X.shape[0])):
        user_focus = df.iloc[i]['userid']
        # index the n most similiar users into the row
        most_similiar_i_users = (-sim_mat[i].A).argsort()[0][1:n+1]
        # map back into the array to obtain the values
        weights = (sim_mat[i].A[0])[most_similiar_i_users].tolist()
        # and index into the dataframe to obtain the user ids
        user_ids = [int(df.iloc[j]['userid']) for j in most_similiar_i_users]
        user_sim[int(user_focus)]['user_id'] = set(user_ids)
        user_sim[int(user_focus)]['user_weight'] = set(weights)
    return user_sim


def build_similiarity_dataframe(path):
    target_columns = ['answers_body', 'asks_body', 'asks_title', 'comments_body']
    user_qac = pd.read_csv(path)
    user_qac.fillna('', inplace=True)

    vectorizer = TfidfVectorizer(stop_words='english', analyzer='word', max_df=.9)
    for column in target_columns:
        X = vectorizer.fit_transform(user_qac[column].values)
        similiarity_graph = build_n_most_similiar_users(X, user_qac)

        with open('user_similarity_network.p', 'wb') as fp:
            pickle.dump(similiarity_graph, fp)


def build_user_similiarity_network():
    build_similiarity_dataframe('../../160-Stackoverflow-Data/300000_rows/user_communication.csv')
