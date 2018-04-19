import progressbar
import csv
from bs4 import BeautifulSoup
import mmap
import numpy as np


def mapcount(filename):
    """
    :param filename: str - path to file
    :return: int - number of lines in file
    """
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines


def tag2row(xml_tag, column_types):
    """
    :param xml_tag: str - single xml line tag
    :param column_types: List[(str, type)] - column types and id
    :return:
    """
    row = BeautifulSoup(xml_tag, "lxml").row
    if not row:
        return None
    csv_row = []
    for column, _type in column_types:
        try:
            csv_row.append(_type(row[column.lower()]))
        except KeyError:
            csv_row.append('')
    return csv_row


def xml2csv(in_path, out_path, column_types, random=(False, 300000), limit=(True, 300000)):
    """
    :param in_path: str - xml infile path
    :param out_path: str - csv outfile path
    :param column_types: List[(str, type)] - column types and id
    :param limit: int - how many rows to parse
    :return:
    """
    with open(out_path, 'w', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        headers = [header for header, _type in column_types]
        csv_writer.writerow(headers)

        bar = progressbar.ProgressBar()
        with open(in_path, 'r', encoding="utf8") as infile:
            # generate random numbers, these will be the rows (noncumulative)
            # that we will extract the data from mean and standard deviation
            if random[0]:
                n_lines = mapcount(in_path)
                mu, sigma = n_lines // random[1], 5
                row_extract = np.random.normal(mu, sigma, random[2]).astype(int)
                row_extract_iter, row_count = 0, 0

                for line in bar(infile):
                    row_count += 1
                    if row_extract_iter < len(row_extract) and row_count == row_extract[row_extract_iter]:
                        # reset to next row extract in order to make cumulative
                        row_extract_iter += 1
                        row_count = 0
                        row = tag2row(line, column_types)
                        if row:
                            csv_writer.writerow(tag2row(line, column_types))
            # read the file sequentially in reverse in order
            # to read the more recent data. This is because the table is sorted
            # by creation date
            elif limit[0]:
                n_lines = mapcount(in_path)
                # if the number of rows we want to read exceeds the number of
                # rows in the file, then we should still be fine
                seek = n_lines - limit[1]
                for line in bar(infile):
                    if seek > 0:
                        seek -= 1
                    else:
                        row = tag2row(line, column_types)
                        if row:
                            csv_writer.writerow(tag2row(line, column_types))


if __name__ == "__main__":
    # Postlinks.xml
    column_types = [('Id', int), ('CreationDate', str), ('PostId', int), ('RelatedPostId', int), ('LinkTypeId', int)]
    xml2csv('Postlinks.xml', 'Postlinks.csv', column_types)


    # Users.xml
    column_types = [('Id', int), ('Reputation', str), ('CreationDate', str), ('DisplayName', str), ('EmailHash', str),
                    ('LastAccessDate', str), ('WebsiteUrl', str), ('Location', str), ('Age', int), ('AboutMe', str),
                    ('Views', int), ('UpVotes', int), ('DownVotes', int)]
    xml2csv('Users.xml', 'Users.csv', column_types)


    # Votes.xml
    column_types = [('Id', int), ('PostId', int), ('VoteTypeId', int), ('CreationDate', str), ('UserId', int),
                    ('BountyAmount', int)]
    xml2csv('Votes.xml', 'Votes.csv', column_types)


    # Posts.xml
    column_types = [('Id', int), ('PostTypeId', int), ('ParentID', int), ('AcceptedAnswerId', int), ('CreationDate', str),
                    ('Score', int), ('ViewCount', int), ('Body', str), ('OwnerUserId', int), ('LastEditorUserId', int),
                    ('LastEditorDisplayName', str), ('LastEditDate', str), ('LastActivityDate', str),
                    ('CommunityOwnedDate', str), ('ClosedDate', str), ('Title', str), ('Tags', str), ('AnswerCount', int),
                    ('CommentCount', int), ('FavoriteCount', int)]
    xml2csv('Posts.xml', 'Posts.csv', column_types)


    # Badges.xml
    column_types = [('UserId', int), ('Name', str), ('Date', str)]
    xml2csv('Badges.xml', 'Badges.csv', column_types)


    # Comments.xml
    column_types = [('Id', int), ('PostId', int), ('Score', int), ('Text', str), ('CreationDate', str), ('UserId', int)]
    xml2csv('Comments.xml', 'Comments.csv', column_types)


    # PostHistory.xml
    column_types = [('Id', int), ('PostHistoryTypeId', int), ('PostId', int), ('RevisionGUID', str), ('CreationDate', str),
                    ('UserId', int), ('UserDisplayName', str), ('Comment', str), ('Text', str), ('CloseReasonId', int)]
    xml2csv('PostHistory.xml', 'PostHistory.csv', column_types)
