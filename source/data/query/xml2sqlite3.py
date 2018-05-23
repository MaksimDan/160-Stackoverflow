import sqlite3
import os
from xml.etree import ElementTree
import logging
import codecs
import progressbar


ANATHOMY = {
    'Badges': {
        'Id': 'INTEGER',
        'UserId': 'INTEGER',
        'Class': 'INTEGER',
        'Name': 'TEXT',
        'Date': 'DATETIME',
        'TagBased': 'BOOLEAN',
    },
    'Comments': {
        'Id': 'INTEGER',
        'PostId': 'INTEGER',
        'Score': 'INTEGER',
        'Text': 'TEXT',
        'CreationDate': 'DATETIME',
        'UserId': 'INTEGER',
        'UserDisplayName': 'TEXT'
    },
    'Posts': {
        'Id': 'INTEGER',
        'PostTypeId': 'INTEGER',  # 1: Question, 2: Answer
        'ParentId': 'INTEGER',  # (only present if PostTypeId is 2)
        'AcceptedAnswerId': 'INTEGER',  # (only present if PostTypeId is 1)
        'CreationDate': 'DATETIME',
        'Score': 'INTEGER',
        'ViewCount': 'INTEGER',
        'Body': 'TEXT',
        'OwnerUserId': 'INTEGER',  # (present only if user has not been deleted)
        'OwnerDisplayName': 'TEXT',
        'LastEditorUserId': 'INTEGER',
        'LastEditorDisplayName': 'TEXT',  # ="Rich B"
        'LastEditDate': 'DATETIME',  # ="2009-03-05T22:28:34.823"
        'LastActivityDate': 'DATETIME',  # ="2009-03-11T12:51:01.480"
        'CommunityOwnedDate': 'DATETIME',  # (present only if post is community wikied)
        'Title': 'TEXT',
        'Tags': 'TEXT',
        'AnswerCount': 'INTEGER',
        'CommentCount': 'INTEGER',
        'FavoriteCount': 'INTEGER',
        'ClosedDate': 'DATETIME'
    },
    'Votes': {
        'Id': 'INTEGER',
        'PostId': 'INTEGER',
        'UserId': 'INTEGER',
        'VoteTypeId': 'INTEGER',
        # -   1: AcceptedByOriginator
        # -   2: UpMod
        # -   3: DownMod
        # -   4: Offensive
        # -   5: Favorite
        # -   6: Close
        # -   7: Reopen
        # -   8: BountyStart
        # -   9: BountyClose
        # -  10: Deletion
        # -  11: Undeletion
        # -  12: Spam
        # -  13: InformModerator
        'CreationDate': 'DATETIME',
        'BountyAmount': 'INTEGER'
    },
    'PostHistory': {
        'Id': 'INTEGER',
        'PostHistoryTypeId': 'INTEGER',
        'PostId': 'INTEGER',
        'RevisionGUID': 'TEXT',
        'CreationDate': 'DATETIME',
        'UserId': 'INTEGER',
        'UserDisplayName': 'TEXT',
        'Comment': 'TEXT',
        'Text': 'TEXT'
    },
    'Postlinks': {
        'Id': 'INTEGER',
        'CreationDate': 'DATETIME',
        'PostId': 'INTEGER',
        'RelatedPostId': 'INTEGER',
        'PostLinkTypeId': 'INTEGER',
        'LinkTypeId': 'INTEGER'
    },
    'Users': {
        'Id': 'INTEGER',
        'Reputation': 'INTEGER',
        'CreationDate': 'DATETIME',
        'DisplayName': 'TEXT',
        'LastAccessDate': 'DATETIME',
        'WebsiteUrl': 'TEXT',
        'Location': 'TEXT',
        'Age': 'INTEGER',
        'AboutMe': 'TEXT',
        'Views': 'INTEGER',
        'UpVotes': 'INTEGER',
        'DownVotes': 'INTEGER',
        'AccountId': 'INTEGER',
        'ProfileImageUrl': 'TEXT'
    },
    'Tags': {
        'Id': 'INTEGER',
        'TagName': 'TEXT',
        'Count': 'INTEGER',
        'ExcerptPostId': 'INTEGER',
        'WikiPostId': 'INTEGER'
    }
}


def dump_files(file_names, anathomy,
               dump_database_name='so.db',
               create_query='CREATE TABLE IF NOT EXISTS {table} ({fields})',
               insert_query='INSERT INTO {table} ({columns}) VALUES ({values})',
               log_filename='so-parser.log'):
    logging.basicConfig(filename=log_filename, level=logging.INFO)
    db = sqlite3.connect(dump_database_name)
    for file in file_names:
        print("Opening {0}.xml".format(file))
        with open(f'{file}.xml', encoding='utf-8') as xml_file:
            tree = iter(ElementTree.iterparse(xml_file))

            table_name = file
            sql_create = create_query.format(
                table=table_name,
                fields=", ".join([f'{name} {_type}' for name, _type in anathomy[table_name].items()]))
            print('Creating table {0}'.format(table_name))

            try:
                logging.info(sql_create)
                db.execute(sql_create)
            except Exception as e:
                logging.warning(e)

            bar = progressbar.ProgressBar()
            for events, row in bar(tree):
                try:
                    if row.attrib.values():
                        logging.debug(row.attrib.keys())
                        query = insert_query.format(
                            table=table_name,
                            columns=', '.join(row.attrib.keys()),
                            values=('?, ' * len(row.attrib.keys()))[:-2])
                        vals = []
                        for key, val in row.attrib.items():
                            if anathomy[table_name][key] == 'INTEGER':
                                vals.append(int(val))
                            elif anathomy[table_name][key] == 'BOOLEAN':
                                vals.append(1 if val == "TRUE" else 0)
                            else:
                                vals.append(val)
                        db.execute(query, vals)

                except Exception as e:
                    logging.warning(e)
                    print("x", end="")
                finally:
                    row.clear()
            print("\n")
            db.commit()
            del tree


def convert_to_utf8(file_names):
    BLOCKSIZE = 1048576
    for file in file_names:
        with codecs.open(f'{file}.xml', "r", "<your input type>") as sourceFile:
            with codecs.open(f'{file}_new.xml', "w", "utf-8") as targetFile:
                while True:
                    contents = sourceFile.read(BLOCKSIZE)
                    if not contents:
                        break
                    targetFile.write(contents)


if __name__ == '__main__':
    # convert_to_utf8(ANATHOMY.keys())
    dump_files(ANATHOMY.keys(), ANATHOMY)

