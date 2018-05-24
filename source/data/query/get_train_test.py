import sqlite3
import pandas as pd
import time

path = 'G:/so.db'
data = sqlite3.connect(path)


# the base query returns all the user activity associated
# between the months january to june.
# Strategy: first find all questions between these dates, then grab all
#           of the posts that link their parentId to this question.
def get_base_query(column_name):
    BASE_QUERY = \
        f"""
        SELECT {column_name}
        FROM Posts
        WHERE (CreationDate BETWEEN date('2012-01-01') AND date('2012-06-30')
               AND (LOWER(Tags) LIKE '%<java>%'))
        UNION ALL
        SELECT {column_name}
        FROM Posts
        WHERE
        ParentID IN (SELECT Id FROM Posts WHERE CreationDate BETWEEN date('2012-01-01')
                     AND date('2012-06-30') AND (LOWER(Tags) LIKE '%<java>%'))
        """
    return BASE_QUERY


#############################################################################

Posts = get_base_query('*')
t1 = time.time()
Posts = pd.read_sql(Posts, data)
Posts.to_csv('Posts.csv', index=False)
t2 = time.time()

print('Posts Done', t2-t1)

#############################################################################

Comments = \
    f"""
    SELECT Comments.*
    FROM Comments
    WHERE Comments.PostId
    IN
    ({get_base_query('Posts.Id')})
    """
t1 = time.time()
Comments_df = pd.read_sql(Comments, data)
Comments_df.to_csv('Comments.csv', index=False)
t2 = time.time()

print('Comments Done', t2 - t1)

#############################################################################

Postlinks = \
    f"""
    SELECT Postlinks.*
    FROM Postlinks
    WHERE Postlinks.PostId
    IN
    ({get_base_query('Posts.Id')})
    """
t1 = time.time()
Postlinks_df = pd.read_sql(Postlinks, data)
Postlinks_df.to_csv('Postlinks.csv', index=False)
t2 = time.time()

print('Postlinks Done', t2 - t1)

#############################################################################

Votes = \
    f"""
    SELECT Votes.*
    FROM Votes
    WHERE Votes.PostId
    IN
    ({get_base_query('Posts.Id')})
    """
t1 = time.time()
Votes_df = pd.read_sql(Votes, data)
Votes_df.to_csv('Votes.csv', index=False)
t2 = time.time()

print('Votes Done', t2 - t1)

#############################################################################

# notice that users can have activity associated within the posts
# and post history
Users = \
    f"""
    SELECT Users.*
    FROM Users
    WHERE Users.Id
    IN ({get_base_query('Posts.OwnerUserId')}
    UNION
    SELECT PostHistory.UserId
    FROM PostHistory
    WHERE PostHistory.UserId
    IN ({get_base_query('Posts.OwnerUserId')}))
    """
t1 = time.time()
Users_df = pd.read_sql(Users, data)
Users_df.to_csv('Users.csv', index=False)
t2 = time.time()

print('Users Done', t2 - t1)

#############################################################################

PostHistory = \
    f"""
    SELECT PostHistory.*
    FROM PostHistory
    WHERE PostHistory.PostId
    IN ({get_base_query('Posts.Id')})
    """
t1 = time.time()
PostHistory_df = pd.read_sql(PostHistory, data)
PostHistory_df.to_csv('PostHistory.csv', index=False)
t2 = time.time()

print('PostHistory Done', t2 - t1)
