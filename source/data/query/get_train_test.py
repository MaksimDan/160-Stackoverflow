import sqlite3
import pandas as pd
import time

path = 'G:/so.db'
data = sqlite3.connect(path)

#############################################################################

def get_base_query(column_name):
    BASE_QUERY = \
        f"""
        SELECT {column_name}
        FROM Posts
        WHERE CreationDate BETWEEN date('2012-01-01') AND date('2012-06-30')
        AND (LOWER(Tags) LIKE '%<java>%')
        UNION ALL
        SELECT {column_name}
        FROM Posts
        WHERE PostTypeId = 2 
        AND 
        ParentID IN (SELECT Id FROM Posts WHERE CreationDate BETWEEN date('2012-01-01')
                   AND date('2012-06-30') AND (LOWER(Tags) LIKE '%<java>%'))
        """
    return BASE_QUERY


# Posts
t1 = time.time()

Posts_2012_Jan_June = get_base_query('*')

Posts_2012_Jan_June = pd.read_sql(Posts_2012_Jan_June, data)
Posts_2012_Jan_June.to_csv('Posts_2012_Jan_June.csv', index=False)

t2 = time.time()
print('Posts Done', t2-t1)

#############################################################################

#Comments
t1 = time.time()

Comments_2012_Jan_June = \
    f"""
    SELECT Comments.*
    FROM Comments
    WHERE Comments.PostId
    IN 
    ({get_base_query('Posts.Id')})
    """
Comments_2012_Jan_June_df = pd.read_sql(Comments_2012_Jan_June, data)
Comments_2012_Jan_June_df.to_csv('Comments_2012_Jan_June.csv', index=False)

t2 = time.time()
print('Comments Done', t2 - t1)

#############################################################################

#Postlinks
t1 = time.time()

Postlinks_2012_Jan_June = \
    f"""
    SELECT Postlinks.*
    FROM Postlinks
    WHERE Postlinks.PostId
    IN 
    ({get_base_query('Posts.Id')})
    """
Postlinks_2012_Jan_June_df = pd.read_sql(Postlinks_2012_Jan_June, data)
Postlinks_2012_Jan_June_df.to_csv('Postlinks_2012_Jan_June.csv', index=False)

t2 = time.time()
print('Postlinks_2012 Done', t2 - t1)

#############################################################################

#Votes
t1 = time.time()

Votes_2012_Jan_June = \
    f"""
    SELECT Votes.*
    FROM Votes
    WHERE Votes.PostId
    IN 
    ({get_base_query('Posts.Id')})
    """
Votes_2012_Jan_June_df = pd.read_sql(Votes_2012_Jan_June, data)
Votes_2012_Jan_June_df.to_csv('Votes_2012_Jan_June.csv', index=False)

t2 = time.time()
print('Votes_2012 Done', t2 - t1)

#############################################################################

#Users
t1 = time.time()

Users_2012_Jan_June = \
    f"""
    SELECT Users.*
    FROM Users
    WHERE Users.Id
    IN 
    ({get_base_query('Posts.OwnerUserId')})
    """
Users_2012_Jan_June_df = pd.read_sql(Users_2012_Jan_June, data)
Users_2012_Jan_June_df.to_csv('Users_2012_Jan_June.csv', index=False)

t2 = time.time()
print('Users_2012 Done', t2 - t1)

#############################################################################

#Posthistory
t1 = time.time()

PostHistory_2012_Jan_June = \
    f"""
    SELECT PostHistory.*
    FROM PostHistory
    WHERE PostHistory.PostId
    IN 
    ({get_base_query('Posts.Id')})
    """
PostHistory_2012_Jan_June_df = pd.read_sql(PostHistory_2012_Jan_June, data)
PostHistory_2012_Jan_June_df.to_csv('PostHistory_2012_Jan_June.csv', index=False)

t2 = time.time()
print('PostHistory_2012 Done', t2 - t1)
