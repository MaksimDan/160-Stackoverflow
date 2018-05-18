import pandas as pd
import json
from collections import defaultdict
import progressbar
import sys
import jsonpickle

"""
File: build_tag_similiarity_network.py
Objective: Represent the cosine relationship between tags.
def build_all_tag_network
Graph Structure:
    {
        <tag1> = [[tag, strength], [tag, strength], ... ],
        <tag2> = [[tag, strength], [tag, strength], ... ],
        ...
        <tagn> = [[tag, strength], [tag, strength], ... ],
    }
"""


def build_all_tag_network(all_tags_path, data_ratio, _indent=False):
    """
    :param all_tags_path: path to AllTags.csv
    :param data_ratio: ratio of data to sample
    @param _indent: int - indent space for printy print
    :return: str - JSON graph
    """
    d = defaultdict(list)
    all_tags = pd.read_csv(all_tags_path)
    data_amount = int(len(all_tags) * data_ratio)
    all_tags = all_tags.sample(data_amount).Tags.values

    bar = progressbar.ProgressBar()
    for i, tags in enumerate(bar(all_tags)):
        for tag in str(tags).split():
            d[str(tag)].append(i)
    return json.dumps(d)


def cosine_sim(tag1, tag2, tag_dict):
    tag1_set = set(tag_dict[tag1])
    tag2_set = set(tag_dict[tag2])
    num =  len(tag1_set & tag2_set)
    den = (len(tag1_set)**2)**5 * (len(tag2_set)**2)**.5
    return num / den 


def build_graph():
    graph = defaultdict(dict)
    for i in range(n):
        tag = all_tags[i]
        similar_tags = tag_similarity.loc[tag,:].nlargest(n=15)
        graph[tag] = {'similar_tags' : [(a, b) for a, b in zip(similar_tags.axes[0].tolist(), similar_tags.values.tolist())\
                                       if b > 0]}
    return graph


def build_network():
    tag_dict = json.load(open('/Users/John/Dropbox/160-Stackoverflow-Data/tags/tag_index_network.json'))
    
    # store the cosine(tag1, tag2)
    all_tags = list(tag_dict)
    n = len(all_tags)
    tag_similarity = pd.DataFrame([],columns=all_tags[:n], index=all_tags[:n])
    for i in range(n-1):
        tag1 = all_tags[i]
        for j in range(i+1, n-1):
            tag2 = all_tags[j]
            tag_similarity.at[tag1,tag2] = cosine_sim(tag1, tag2, tag_dict)

    tag_similarity = tag_similarity.astype(float)
    graph = build_graph()
    with open('tag_network_graph.json', 'w+') as outfile:
        outfile.write(jsonpickle.encode(graph))

if __name__ == "__main__":
    print(build_all_tag_network('/Users/John/Dropbox/160-Stackoverflow-Data/tags/AllTagsClean.csv', 2/3))
    print(build_network())
