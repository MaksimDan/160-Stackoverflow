import pandas as pd
from collections import defaultdict
import json
import progressbar
import sys

Posts = pd.read_csv('../../160-Stackoverflow-Data/train_test/raw/Posts_2012_Clean.csv')
Comments = pd.read_csv('../../160-Stackoverflow-Data/train_test/Comments.csv')

# first need to have the answers dataframe contain the tags from the question
# we will also need to merge the post id of a comment with this as well
# this dataframe below will be used for that purpose
post_answers_merge = Posts.loc[Posts.PostTypeId == 1][['Tags', 'Id']]

user_profile = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: float())))

bar = progressbar.ProgressBar()
for key, group in bar(Posts.groupby('OwnerUserId')):
    user_q = group[group.PostTypeId == 1]
    user_a = group[group.PostTypeId == 2]
    user_c = Comments.loc[Comments.UserId == int(key)]

    # now tally up the number of questions per tag
    if len(user_q) != 0:
        for index, row in user_q.iterrows():
            try:
                for tag in row.Tags.split():
                    user_profile[key][tag]['n_questions'] += 1
            except AttributeError as e:
                continue

    # tally up the number of answers per tag
    user_a = pd.merge(user_a, post_answers_merge, left_on='ParentId', right_on='Id', how='left')
    if len(user_a) != 0:
        for index, row in user_a.iterrows():
            try:
                for tag in row.Tags_y.split():
                    user_profile[key][tag]['n_answers'] += 1
            except AttributeError as e:
                continue

    # tally up the number of comments per tag
    user_c = pd.merge(user_c, post_answers_merge, left_on='PostId', right_on='Id', how='left')
    if len(user_c) != 0:
        for index, row in user_c.iterrows():
            try:
                for tag in row.Tags.split():
                    user_profile[key][tag]['n_comments'] += 1
            except AttributeError as e:
                continue

# finally, save as json
sys.stdout = open("user_expertise.json", "w")
print(json.dumps(user_profile))
