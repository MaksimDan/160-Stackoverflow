import sqlite3
import os

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

def create_tables():
	cursor = connection.cursor()
	table_names = ['badges', 'comments', 'posts', 'posthistory', 'postlinks', 'users', 'votes', 'tags']
	create_tables = [
					 f'CREATE TABLE {table_names[0]} (UserId int, Name text, Date text)',
					 f'CREATE TABLE {table_names[7]} (Id int, TagName text, Count int, ExcerptPostId int, WikiPostId int)'             
					 f'CREATE TABLE {table_names[1]} (Id int, PostId int, Score int, Text text, CreationDate text, UserId int)'
					 f'CREATE TABLE {table_names[2]} (UserId int, Name text, Date text)'
					 f'CREATE TABLE {table_names[3]} (UserId int, Name text, Date text)'
					 f'CREATE TABLE {table_names[4]} (UserId int, Name text, Date text)'
					 f'CREATE TABLE {table_names[5]} (UserId int, Name text, Date text)'
					 f'CREATE TABLE {table_names[6]} (UserId int, Name text, Date text)'
					]
	
	for name, create in zip(table_names, create_tables):
		connection = sqlite3.connect(DIR_PATH + f'\\{name}.db')
		cursor.execute(create)
		connection.commit()
	connection.close()


if __name__ == "__main__":
	create_tables()


# thoughts
# work with reasonably random sample from all stackoverflow data
#   have these written as a csv
#   later work with sqlite databases for the complete files