import pandas as pd
import json
import sys
import progressbar
from collections import defaultdict

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
