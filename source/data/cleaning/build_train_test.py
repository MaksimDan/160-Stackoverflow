import pandas as pd
import re
from sklearn.model_selection import train_test_split
from copy import copy
import progressbar

# raw input
Posts_full = pd.read_csv('../../160-Stackoverflow-Data/train_test/raw/Posts_2012.csv')
Posts_full.dropna(subset=['OwnerUserId'], inplace=True)
Posts_full['OwnerUserId'] = Posts_full['OwnerUserId'].astype('int')

# build X
Posts_X = Posts_full.loc[Posts_full.PostTypeId == 1]
X = Posts_X.drop(columns=['PostTypeId', 'LastEditorDisplayName', 'LastEditDate', 'LastActivityDate', 'CommunityOwnedDate'])
X.Tags = X.Tags.apply(lambda t: ' '.join(re.findall(r"<(\w+)>", str(t))))
X.to_csv('X.csv', index=False)

# build y
answerers_per_question = Posts_full.groupby('ParentId')
question_and_answerers = {questionid: ' '.join(str(x) for x in answerers['OwnerUserId'].values) for questionid, answerers in answerers_per_question}

y_list = [question_and_answerers.get(question_id, '') for question_id in X.Id.values]
y = pd.DataFrame({'owner_user_ids': y_list})
y.to_csv('y.csv', index=False)


# map in the synoymns
def remap_tag_synoymns(tag_syn_path, post_inpath):
    tag_syn_df = pd.read_csv(tag_syn_path)
    tag_map = {str(row['SourceTagName']).lower(): str(row['TargetTagName']).lower()
               for index, row in tag_syn_df.iterrows()}

    posts_df = pd.read_csv(post_inpath)
    old_tag_list = posts_df.Tags.values
    new_tag_list = []

    bar = progressbar.ProgressBar()
    for tags in bar(old_tag_list):
        if isinstance(tags, str):
            new_tag_list.append(
                ' '.join([tag_map[tag.lower()] if tag.lower() in tag_map else tag for tag in tags.split()]))
        else:
            new_tag_list.append('')
    posts_df['Tags'] = new_tag_list
    posts_df.to_csv('X.csv', index=False)


remap_tag_synoymns('../../160-Stackoverflow-Data/tags/TagSynonyms.csv', 'X.csv')

# now remove questions that dont have answers, and create the training and test split
X['y'] = y
X.dropna(subset=['y'], inplace=True)
y = pd.DataFrame({'owner_user_ids': copy(X.y)})
X = X.drop('y', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .15, random_state = 42)

# save to csv
X_train.to_csv('X_train.csv')
X_test.to_csv('X_test.csv')
y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')


# build answers.csv
Answers = Posts_full.loc[Posts_full.PostTypeId == 2]

features = ['Id', 'ParentId', 'CreationDate', 'Score', 'ViewCount', 'Body', 'OwnerUserId', 'ClosedDate']
Answers = Answers[features]
Answers.to_csv('Answers.csv', index=False)