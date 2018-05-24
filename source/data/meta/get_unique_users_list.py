import pickle
import pandas as pd

X = pd.read_csv('../../160-Stackoverflow-Data/train_test/raw_query/Users.csv')
X.dropna(subset=['Id'], inplace=True)
X['OwnerUserId'] = X['Id'].astype('int')
unique_users = list(set(X.OwnerUserId.values))
print(len(unique_users))

with open('users_list.p', 'wb') as fp:
    pickle.dump(unique_users, fp)