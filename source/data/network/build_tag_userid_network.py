import pandas as pd
import json
from collections import defaultdict
import progressbar
import sys
from math import isnan
import re


"""
File: tag_user_network.py
Objective: Maps all the users associated with tag_i.
           Outputs a json graph network.

Graph Structure:
    graph = {
        <tag1> = [user_id1, user_id2],
        <tag2> = [user_id1, user_id2],
        
        ...
        
        <tagn> = [user_id1, user_id2],
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


def build_tag_network(_indent=False):
    """
    Builds a network that represents the communication between users.
    @param _indent: int - indent space for printy print
    :return: str - JSON graph
    """
    # load data
    Posts = pd.read_csv('Posts.csv', dtype={'LastEditorDisplayName': str})
    Posts['Tags'] = Posts['Tags'].apply(lambda t: ' '.join(re.findall(r"<(\w+)>", str(t))))

    # initialize dictionary
    d = defaultdict(set)
    bar = progressbar.ProgressBar()

    # add questions and answers
    for post in bar(Posts[['Tags', 'Id']].itertuples()):
        # ensure that user exists
        if isnan(post.Id):
            continue
        tags, owner_id = post.Tags.split(), int(post.Id)
        for tag in tags:
            d[tag].add(owner_id)

    if _indent:
        return json.dumps(d, indent=4, cls=SetEncoder)
    else:
        return json.dumps(d, cls=SetEncoder)


if __name__ == "__main__":
    sys.stdout = open("tag_network_view.json", "w")
    print(build_tag_network(True))
    
    sys.stdout = open("tag_network_load.json", "w")
    print(build_tag_network(False))
