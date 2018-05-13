import pandas as pd
import json
import sys
import progressbar
from collections import defaultdict

"""
File: build_user_basic_profile_network.py
Objective: Build a json file that identifies the basic
		   meta profile properties associated with a user.

Graph Structure:
    {
        <user_id1> = {'prop1': val1, 'prop2': val2, ... },
        <user_id1> = {'prop1': val1, 'prop2': val2, ... },
        ...
        <user_idn> = {'prop1': val1, 'prop2': val2, ... },
    }
"""

# load data
Users = pd.read_csv('../../160-Stackoverflow-Data/train_test/Users.csv')

# only interested in the user and the creation dates
user_basic_profile = defaultdict(lambda: defaultdict(lambda: int()))

bar = progressbar.ProgressBar()
for index, row in bar(Users.iterrows()):
    user_basic_profile[row['Id']]['reputation'] = row.Reputation
    user_basic_profile[row['Id']]['views'] = row.Views
    user_basic_profile[row['Id']]['creation_date'] = row.CreationDate
    user_basic_profile[row['Id']]['upvotes'] = row.UpVotes
    user_basic_profile[row['Id']]['downvotes'] = row.DownVotes

sys.stdout = open("user_basic_profile_network.json", "w")
print(json.dumps(user_basic_profile))
