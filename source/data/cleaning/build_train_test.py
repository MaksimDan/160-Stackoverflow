import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict

# read inputs
BASE_PATH = '../../160-Stackoverflow-Data/train_test/raw_query/'

Posts_full = pd.read_csv(BASE_PATH + 'Posts.csv')
Votes = pd.read_csv(BASE_PATH + 'Votes.csv')
Comments = pd.read_csv(BASE_PATH + 'Comments.csv')
Post_his = pd.read_csv(BASE_PATH + 'PostHistory.csv')

# read inputs
print("Finished Reading Files")

# clean data
Posts_full.dropna(subset=['OwnerUserId'], inplace=True)
Posts_full['OwnerUserId'] = Posts_full['OwnerUserId'].astype('str')


Votes['UserId'] = Votes['UserId'].astype('str')

Comments.dropna(subset=['UserId'], inplace=True)
Comments['UserId'] = Comments['UserId'].astype('str')

Post_his.dropna(subset=['UserId'], inplace=True)
Post_his['UserId'] = Post_his['UserId'].astype('str')
print("Finished Cleaning")

# build X - the questions and features
Posts_X = Posts_full.loc[Posts_full.PostTypeId == 1]
X = Posts_X.drop(columns=['PostTypeId', 'LastEditorDisplayName', 'LastEditDate', 'LastActivityDate', 'CommunityOwnedDate'])
X['Id'] = X['Id'].astype(str)
print("Finished Building X")

# build y - the answers, comments, upvotes, downvotes, favorites
activity = defaultdict(lambda :{'1' : set(), '5': set(), 'editers' : set(), 'commenters': set(), 'answerers': set()})

# answers
Posts_full['ParentId'] = Posts_full['ParentId'].astype(str).apply(lambda row : row.rstrip(".0"))
Posts_full[Posts_full['PostTypeId'] == 2].apply(lambda row : activity[row.ParentId]['answerers'].add(row.OwnerUserId), axis=1)

# comments
Comments.apply(lambda row: activity[str(row.PostId)]['commenters'].add(row.UserId), axis=1)

# vote activites
Votes = Votes[Votes['VoteTypeId'].isin([1, 5])]
Votes.apply(lambda row : activity[str(row.PostId)][str(row.VoteTypeId)].add(row.UserId), axis=1)

# edits
Post_his = Post_his[(Post_his['PostHistoryTypeId'] >= 4) & (Post_his['PostHistoryTypeId'] <= 6)]
Post_his.apply(lambda row: activity[str(row.PostId)]['editers'].add(row.UserId), axis=1)


d = {'favorite': [], 'editers' : [], 'commenters': [], 'answerers': []}
for x in activity.values():
    d['favorite'].append([int(float(user)) for user in list(x['5'])])
    d['editers'].append([int(float(user)) for user in list(x['editers'])])
    d['commenters'].append([int(float(user)) for user in list(x['commenters'])])
    d['answerers'].append([int(float(user)) for user in list(x['answerers'])])

y = pd.DataFrame(d)
y['Id'] = list(activity.keys())
y.set_index('Id')
X.set_index('Id')
full = X.merge(y, on='Id')

y = full.iloc[:,17:21]
y['Id'] = full.iloc[:, 0:1]

X = full.iloc[:, :17]
print("Finished Building Y")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .15, random_state = 42)

# save to csv
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)


