import pandas as pd
import re
from sklearn.model_selection import train_test_split
from copy import copy
import progressbar

# raw input
Posts_full = pd.read_csv('../../160-Stackoverflow-Data/train_test/Posts_Clean.csv')
Posts_full.dropna(subset=['OwnerUserId'], inplace=True)
Posts_full['OwnerUserId'] = Posts_full['OwnerUserId'].astype('int')

# build X - the questions and features
Posts_X = Posts_full.loc[Posts_full.PostTypeId == 1]
X = Posts_X.drop(columns=['PostTypeId', 'LastEditorDisplayName', 'LastEditDate', 'LastActivityDate', 'CommunityOwnedDate'])
X.to_csv('X.csv', index=False)

# build y - the answers, comments, upvotes, downvotes, favorites
answerers_per_question = Posts_full.groupby('ParentId')
questionid_to_answerers = {questionid: list(answerers['OwnerUserId'].values )
                              for questionid, answerers in answerers_per_question}

Comments = pd.read_csv('../../160-Stackoverflow-Data/train_test/Comments.csv')
commenters_per_question = Comments.groupby('PostId')
questionid_to_commentors = {questionid: list(commenters['UserId'].values )
                              for questionid, commenters in commenters_per_question}

# Votes = pd.read_csv('../../160-Stackoverflow-Data/train_test/Votes.csv')
# up_down_votes_per_question = Votes.groupby('PostId')
# questionid_to_voters = {questionid: {'upvoters': list(voters['UserId'].values ), 'downvoters':}
#                               for questionid, voters in up_down_votes_per_question}

y_list = [{'answerers': questionid_to_answerers.get(question_id, ''),
           'commentors': questionid_to_commentors.get(question_id, '')} 
          for question_id in X.Id.values]
y = pd.DataFrame({'owner_user_ids': y_list})
y.to_csv('y.csv', index=False)


# now remove questions that dont have answers, and create the training and test split
X['y'] = y
X.dropna(subset=['y'], inplace=True)
y = pd.DataFrame({'owner_user_ids': copy(X.y)})
X = X.drop('y', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .15, random_state = 42)

# save to csv
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
