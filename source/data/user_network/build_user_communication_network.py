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
File: build_user_post_network.py
Purpose: Builds nested graph that representations the communication
         between users, who ask questions, answer questions, and post
         comments.
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
    no_tags = BeautifulSoup(raw_html, 'lxml').text
    no_tags_and_stopwords = ' '.join([w for w in word_tokenize(no_tags) if w not in STOPWORDS])
    return ' '.join(TextBlob(no_tags_and_stopwords).noun_phrases)


def build_user_communication_df(_indent=False):
    """
    Builds a network that represents the communication between users.
    @param _indent: int - indent space for printy print
    :return: str - JSON graph
    """
    Posts = pd.read_csv('../../160-Stackoverflow-Data/100000_rows/Posts.csv',
                        dtype={'LastEditorDisplayName': str})
    Comments = pd.read_csv('../../160-Stackoverflow-Data/100000_rows/Comments.csv',
                           dtype={'LastEditorDisplayName': str})

    # initialize nested dictionary
    graph = defaultdict(lambda: defaultdict(lambda: list()))
    bar = progressbar.ProgressBar()

    # add questions and answers
    for post in bar(Posts[['PostTypeId', 'OwnerUserId', 'Body', 'Title']].itertuples()):
        # poster asks a question
        if post.PostTypeId == 1:
            try:
                graph[int(post.OwnerUserId)]['asks_body'].append(clean_html(post.Body))
                graph[int(post.OwnerUserId)]['asks_title'].append(clean_html(post.Title))
            except ValueError:
                continue

        # poster answers a question
        elif post.PostTypeId == 2:
            try:
                graph[int(post.OwnerUserId)]['answers_body'].append(clean_html(post.Body))
            except ValueError:
                continue

    # add comments
    for comment in bar(Comments[['UserId', 'Text']].itertuples()):
        try:
            # poster adds a comment
            graph[int(comment.UserId)]['comments_body'].append(clean_html(comment.Text))
        except ValueError:
            continue

    # flatten lists
    second_keys = ['asks_body', 'asks_title', 'answers_body', 'comments_body']
    graph_two = defaultdict(lambda: defaultdict(lambda: str()))
    for user_id in graph.keys():
        for key_two in second_keys:
            graph_two[user_id][key_two] = ' '.join(graph[user_id][key_two])

    test = pd.DataFrame(graph_two)
    test.to_csv('test100k.csv')


if __name__ == "__main__":
    build_user_communication_df()
