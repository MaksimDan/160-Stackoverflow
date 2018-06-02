import pandas as pd

from collections import defaultdict
from bs4 import BeautifulSoup

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import progressbar
from textblob import TextBlob

nltk.download('stopwords')
nltk.download('punkt')


"""
File: build_user_communication_network.py
Objective: Builds nested graph that representations that literally
           concatenates all the postal information by a user.
		   Outputs a json graph network.

Graph Structure:
    graph = {
        <user_id1> = {
           'asks_body': Post1 + Post2 + ... PostN,
           'asks_title': Post1 + Post2 + ... PostN,
           'answers_body': Post1 + Post2 + ... PostN,
           'comments_body': Post1 + Post2 + ... PostN
        },
        <user_id2> = {
           'asks_body': Post1 + Post2 + ... PostN,
           'asks_title': Post1 + Post2 + ... PostN,
           'answers_body': Post1 + Post2 + ... PostN,
           'comments_body': Post1 + Post2 + ... PostN
        },
        ...
        <user_idn> = {
           'asks_body': Post1 + Post2 + ... PostN,
           'asks_title': Post1 + Post2 + ... PostN,
           'answers_body': Post1 + Post2 + ... PostN,
           'comments_body': Post1 + Post2 + ... PostN
        }  
    }
"""

STOPWORDS = set(stopwords.words('english'))


def clean_html(raw_html):
    """
    filters html: removes tags, stopwords, and then extracts nouns
    @param text: str - raw string
    return: str - filtered, and tokenized string
    """
    if not raw_html:
        return ''
    no_tags = BeautifulSoup(raw_html, 'lxml').text
    no_tags_and_stopwords = ' '.join([w for w in word_tokenize(no_tags) if w not in STOPWORDS])
    return ' '.join(TextBlob(no_tags_and_stopwords).noun_phrases)


def build_user_communication_df(base_path, _indent=False):
    """
    Builds a network that represents the communication between users.
    @param _indent: int - indent space for printy print
    :return: str - JSON graph
    """
    from build_all_features import BASE_PATH
    Posts = pd.read_csv(BASE_PATH + 'raw_query/Posts.csv', dtype={'LastEditorDisplayName': str})
    Comments = pd.read_csv(BASE_PATH + 'raw_query/Comments.csv', dtype={'LastEditorDisplayName': str})

    # initialize nested dictionary
    graph = defaultdict(lambda: defaultdict(lambda: str()))

    # add questions and answers
    Posts = Posts[['PostTypeId', 'OwnerUserId', 'Body', 'Title']].dropna(subset=['OwnerUserId'])
    Comments = Comments[['UserId', 'Text']].dropna(subset=['UserId'])
    Questions, Answers = Posts.loc[Posts.PostTypeId == 1], Posts.loc[Posts.PostTypeId == 2]

    print(f'Iterating through {len(Questions)} rows...')
    bar = progressbar.ProgressBar()
    for index, row in bar(Questions.iterrows()):
        graph[int(row.OwnerUserId)]['asks_body'] += ' ' + clean_html(row.Body)
        graph[int(row.OwnerUserId)]['asks_title'] += ' ' + clean_html(row.Title)

    print(f'Iterating through {len(Answers)} rows...')
    bar = progressbar.ProgressBar()
    for index, row in bar(Answers.iterrows()):
        graph[int(row.OwnerUserId)]['answers_body'] += ' ' + clean_html(row.Body)

    # add comments
    print(f'Iterating through {len(Comments)} rows...')
    bar = progressbar.ProgressBar()
    for index, row in bar(Comments.iterrows()):
        graph[int(row.UserId)]['comments_body'] += ' ' + clean_html(row.Text)

    # transform into friendly csv
    pd.DataFrame(graph).to_csv('user_communication.csv')

    # read im as csv and transform to proper friendly format
    user_qac = pd.read_csv('user_communication.csv').T
    headers = user_qac.iloc[0]
    user_qac = user_qac[1:]
    user_qac.columns = headers
    user_qac.insert(0, 'userid', user_qac.index)
    user_qac.reset_index(inplace=True)
    user_qac.drop('index', axis=1, inplace=True)
    user_qac.to_csv('user_communication.csv')

