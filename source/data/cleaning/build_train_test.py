import pandas as pd
import re
from sklearn.model_selection import train_test_split
from copy import copy
import json 
from collections import defaultdict


# read inputs
Posts_full = pd.read_csv('~/Dropbox/160-Stackoverflow-Data/train_test/Posts_Clean.csv')
Votes = pd.read_csv('~/Dropbox/160-Stackoverflow-Data/train_test/Votes.csv')
Comments = pd.read_csv('~/Dropbox/160-Stackoverflow-Data/train_test/Comments.csv')
Post_his = pd.read_csv('~/Dropbox/160-Stackoverflow-Data/train_test/PostsHistory_2012.csv')

# clean data 
Posts_full.dropna(subset=['OwnerUserId'], inplace=True)
Posts_full['OwnerUserId'] = Posts_full['OwnerUserId'].astype('str')

Votes.dropna(subset=['UserId'], inplace=True)
Votes['UserId'] = Votes['UserId'].astype('str')

Comments.dropna(subset=['UserId'], inplace=True)
Comments['UserId'] = Comments['UserId'].astype('str')

Post_his.dropna(subset=['UserId'], inplace=True)
Post_his['UserId'] = Post_his['UserId'].astype('str')


# build X - the questions and features
Posts_X = Posts_full.loc[Posts_full.PostTypeId == 1]
X = Posts_X.drop(columns=['Unnamed: 0','PostTypeId', 'LastEditorDisplayName', 'LastEditDate', 'LastActivityDate', 'CommunityOwnedDate'])
X['Id'] = X['Id'].astype(str) 
X.to_csv('X.csv', index=False)


# build y - the answers, comments, upvotes, downvotes, favorites 
activity = defaultdict(lambda :{'1' : [], '2': [], '3': [], '5': [], 'editers' : [], 'commenters': [], 'answerers': []})

# answers
Posts_full['ParentId'] = Posts_full['ParentId'].astype(str).apply(lambda row : row.rstrip(".0"))
Posts_full[Posts_full['PostTypeId'] == 2].apply(lambda row : activity[row.ParentId]['answerers'].append(row.OwnerUserId), axis=1)

# comments
Comments.apply(lambda row: activity[row.PostId]['commenters'].append(row.UserId), axis=1)

# vote activites
Votes = Votes[Votes['VoteTypeId'].isin([1, 2, 3, 5])]
Votes.apply(lambda row : activity[row.PostId][str(row.VoteTypeId)].append(row.UserId), axis=1)

# edits
Post_his = Post_his[(Post_his['PostHistoryTypeId'] >= 4) & (Post_his['PostHistoryTypeId'] <= 6)]    
Post_his.apply(lambda row: activity[row.PostId]['editers'].append(row.UserId), axis=1)


y = pd.DataFrame.from_dict(activity, orient='index')
y = y.rename(index=str, columns={"1": "accepted", "2": "upvotes", "3": "downvotes", "5": "favorite"})

# split to train test
y.to_csv('y.csv', index_label='Id')
X = X.set_index('Id')
full = X.join(y)
y = full.iloc[:,16:24]
X = full.iloc[:,0:15]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .15, random_state = 42)

# save to csv
X_train.to_csv('X_train.csv', index_label='Id')
X_test.to_csv('X_test.csv', index_label='Id')
y_train.to_csv('y_train.csv', index_label='Id')
y_test.to_csv('y_test.csv', index_label='Id')