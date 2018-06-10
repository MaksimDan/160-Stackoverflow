import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import dill as pickle
import progressbar
from scipy.sparse import save_npz
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

# nltk.download('stopwords')
# nltk.download('punkt')
STOPWORDS = set(stopwords.words('english'))


"""
File: build_user_question_similarity_matrix.py
Objective: Identifies the similarity strength between question_i and user_j
"""


def clean_html(raw_html):
    """
    filters html: removes tags, stopwords, and then extracts nouns
    @param text: str - raw string
    return: str - filtered, and tokenized string
    """
    if not raw_html:
        return ''
    no_tags = BeautifulSoup(raw_html, 'lxml').text
    no_tags_and_stopwords = ' '.join([w for w in word_tokenize(no_tags) if w not in STOPWORDS])
    return ' '.join(TextBlob(no_tags_and_stopwords).noun_phrases)


def build_user_question_similarity_matrix():
    from build_all_features import BASE_PATH

    X_train = pd.read_csv(BASE_PATH + 'X_train.csv').head(800)
    user_qac = pd.read_csv(BASE_PATH + 'meta/user_communication.csv')
    user_qac.fillna('', inplace=True)

    # for the question, combine the question title and the body. Give more weight
    # to the title because it is more important
    weight = 8
    X_train['Title+Body_filtered'] = X_train.apply(lambda row: (row['Title']*weight) +
                                                                clean_html(row['Body']), axis=1)

    # set maximum features to 1000 to not overkill
    bar = progressbar.ProgressBar()
    tf = TfidfVectorizer(stop_words='english', analyzer='word', max_features=1000)
    for column in bar(['answers_body', 'comments_body', 'asks_body', 'asks_title']):
        content_and_history = list(X_train['Title+Body_filtered'].values) + list(user_qac[column].values)
        tf_M = tf.fit_transform(content_and_history)
        M = (tf_M * tf_M.T)[0:X_train.shape[0], X_train.shape[0]:len(content_and_history)]
        save_npz(f'{column}.npz', M)

    # in order to become able to key into the matrix by q_id (row), and u_id (col)
    # add column and row key in (respectively)
    # note: even though the row mapping is obvious, it is good better practice
    #       to hide this information to the person using it
    matrix_key_in = {}
    row_key_in = {i: i for i in range(len(X_train))}
    column_key_in = {userid: j for j, userid in enumerate(user_qac['userid'].values)}
    matrix_key_in['q_to_row'] = row_key_in
    matrix_key_in['user_to_col'] = column_key_in
    with open('key.p', 'wb') as fp:
        pickle.dump(matrix_key_in, fp)

