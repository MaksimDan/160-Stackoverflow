import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import progressbar
import sys
import json
from collections import defaultdict


def build_n_most_similiar_users(X, df, n=15):
    user_sim = defaultdict(lambda: defaultdict(lambda: list()))
    # we need both full loops because also want to establish the
    # relationship the inverse relationship in the graph
    bar = progressbar.ProgressBar()
    for i in bar(range(X.shape[0])):
        user_sim_i = []
        for j in range(X.shape[0]):
            if i != j:
                similarity = (X[i] * X[j].T).A[0][0]
                if similarity > 0:
                    user_sim_i.append((df.iloc[j]['userid'], similarity))
        # append the n most similiar j users (along with weight) to user i
        most_similar = sorted(user_sim_i, key=lambda x: x[1], reverse=True)[0:n]
        user_sim[i]['user_id'] = [int(id) for id, _ in most_similar]
        user_sim[i]['user_weight'] = [float(weight) for _, weight in most_similar]
    return user_sim


def build_similiarity_dataframe(inpaths):
    target_columns = ['answers_body', 'asks_body', 'asks_title', 'comments_body']
    for path in inpaths:
        user_qac = pd.read_csv(path)
        user_qac.fillna('', inplace=True)

        vectorizer = TfidfVectorizer(stop_words='english', analyzer='word', max_df=.9)
        for column in target_columns:
            X = vectorizer.fit_transform(user_qac[column].values)
            similiarity_graph = build_n_most_similiar_users(X, user_qac)
            sys.stdout = open(f"{column}", "w")
            print(json.dumps(similiarity_graph, indent=4))


if __name__ == "__main__":
    inpaths = ['../../160-Stackoverflow-Data/2500_rows/user_communication.csv']
               # '../../160-Stackoverflow-Data/300000_rows/user_communication.csv']
    build_similiarity_dataframe(inpaths)
