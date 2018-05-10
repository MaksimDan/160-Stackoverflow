import pandas as pd
import json
import progressbar
import sys


# load data
Comments = pd.read_csv('../../160-Stackoverflow-Data/train_test/Comments.csv')
Posts = pd.read_csv('../../160-Stackoverflow-Data/train_test/raw/Posts_2012_Clean.csv')
Votes = pd.read_csv('../../160-Stackoverflow-Data/train_test/Votes.csv')

# only interested in the user and the creation dates
Full = Comments[['UserId', 'CreationDate']].append(Posts[['OwnerUserId', 'CreationDate']]).append(Votes[['UserId', 'CreationDate']])
Full['CreationDate'] = pd.to_datetime(Full['CreationDate'], format="%Y-%m-%dT%H:%M:%S")

user_availability = {}
bar = progressbar.ProgressBar()
for key, group in bar(Full.groupby('OwnerUserId')):
    activities = group['CreationDate'].dt.hour.value_counts(normalize=True)
    user_availability[int(key)] = activities.to_dict()

sys.stdout = open("user_availibility.json", "w")
print(json.dumps(user_availability))
