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

    X_train = pd.read_csv(BASE_PATH + 'X_train.csv').head(100)
    user_qac = pd.read_csv(BASE_PATH + 'raw_query/user_communication.csv')
    user_qac.fillna('', inplace=True)

    # for the question, combine the question title and the body. Give more weight
    # to the title because it is more important
    weight = 8
    X_train['Title+Body_filtered'] = X_train.apply(lambda row: (row['Title']*weight) +
                                                                clean_html(row['Body']), axis=1)

    # this snippet is used to get a feel for how long this will take
    # for index, row in X_train.iterrows():
    #     a = (row['Title'] * weight) + clean_html(row['Body'])
    #     print(index/len(X_train))

    # set maximum features to 1000 to not overkill
    vectorizer = TfidfVectorizer(stop_words='english', analyzer='word', max_features=10)
    bar = progressbar.ProgressBar()
    for column in bar(['answers_body', 'comments_body', 'asks_body', 'asks_title']):
        tfidf_questions_and_user_history = vectorizer.fit_transform(list(X_train['Title+Body_filtered'].values) +
                                                                    list(user_qac[column].values))
        M = tfidf_questions_and_user_history * tfidf_questions_and_user_history.T
        print(M.shape)
        print(0, X_train.shape[0], X_train.shape[0], M.shape[1])
        save_npz(f'{column}_question_user_sim_matrix.npz', M)
        # print(M.shape)
        # save_npz(f'{column}_question_user_sim_matrix.npz', M[0, X_train.shape[0]: X_train.shape[0], M.shape[1]])

    # in order to become able to key into the matrix by q_id (row), and u_id (col)
    # add column and row key in (respectively)
    # note: even though the row mapping is obvious, it is good better practice
    #       to hide this information to the person using it
    matrix_key_in = {}
    row_key_in = {i: i for i in range(len(X_train))}
    column_key_in = {userid: j for j, userid in enumerate(user_qac['userid'].values)}
    matrix_key_in['q_to_row'] = row_key_in
    matrix_key_in['user_to_col'] = column_key_in
    with open('user_question_similarity_key_in.p', 'wb') as fp:
        pickle.dump(matrix_key_in, fp)

