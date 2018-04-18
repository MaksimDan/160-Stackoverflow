import pandas as pd
import json
from collections import defaultdict
import progressbar
import sys
from math import isnan


"""
File: build_user_network.py

Purpose: Builds nested graph that representations the communication
         between users, who ask questions, answer questions, and post
         comments.

Graph Structure:
    graph = {
        <user_id1> = {
           'asks': [PostId1, PostId2, ...],
           'answers': [PostId1, PostId2, ...],
           'comments': [PostId1, PostId2, ...]
        },
        <user_id2> = {
           'asks': [PostId1, PostId2, ...],
           'answers': [PostId1, PostId2, ...],
           'comments': [PostId1, PostId2, ...]
        },

        ...

        <user_idn> = {
           'asks': [PostId1, PostId2, ...],
           'answers': [PostId1, PostId2, ...],
           'comments': [PostId1, PostId2, ...]
        }  
    }
"""


class SetEncoder(json.JSONEncoder):
    """
    Converts set into list for JSON encoding
    """
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def build_graph(_indent=False):
    """
    Builds a network that represents the communication between users.
	@param _indent: int - indent space for printy print
    :return: str - JSON graph
    """
    # load data
    Posts = pd.read_csv('../../160-Stackoverflow-Data/300000_rows_[504_MB]/Posts.csv')
    Comments = pd.read_csv('../../160-Stackoverflow-Data/300000_rows_[504_MB]/Comments.csv')

    # initialize nested dictionary
    graph = defaultdict(lambda: defaultdict(lambda: set()))
    bar = progressbar.ProgressBar()

    # add questions and answers
    for post in bar(Posts[['PostTypeId', 'OwnerUserId', 'Id']].itertuples()):
        # validify data
        if any([isnan(p) for p in post]):
            continue
        post_type, owner_id, post_id = int(post.PostTypeId), int(post.OwnerUserId), int(post.Id)
        # poster asks a question
        if post_type == 1:
            graph[owner_id]['asks'].add(post_id)
        # poster answers a question
        elif post_type == 2:
            graph[owner_id]['answers'].add(post_id)

    # add comments
    for comment in bar(Comments[['PostId', 'UserId']].itertuples()):
        # validify data
        if any([isnan(c) for c in comment]):
            continue
        # poster adds a comment
        owner_id, post_id = int(comment.PostId), int(comment.UserId)
        graph[owner_id]['comments'].add(post_id)
    if _indent:
        return json.dumps(graph, indent=4, cls=SetEncoder)
    else:
        return json.dumps(graph, cls=SetEncoder)


if __name__ == "__main__":
    sys.stdout = open("graph_dump_view.txt", "w")
    print(build_graph(True))
    
    sys.stdout = open("graph_dump_load.txt", "w")
    print(build_graph(False))
