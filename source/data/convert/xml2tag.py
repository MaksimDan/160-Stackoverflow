import progressbar
import csv
from bs4 import BeautifulSoup
import re


def tag2row(xml_tag):
    """
    :param xml_tag: str - single xml line tag
    :return:
    """
    row = BeautifulSoup(xml_tag, "lxml").row
    if not row:
        return None
    csv_row = []
    try:
        _xml = str(row['Tags'.lower()])
        csv_row.append(' '.join(re.findall(r"\<(\w+)\>", _xml)))
    except KeyError:
        return None
    return csv_row


def xml2tag(in_path, out_path):
    """
    :param in_path: str - xml infile path
    :param out_path: str - csv outfile path
    :return:
    """
    with open(out_path, 'w', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        csv_writer.writerow(['Tags'])

        bar = progressbar.ProgressBar()
        with open(in_path, 'r', encoding="utf8") as infile:
            for line in bar(infile):
                row = tag2row(line)
                if row:
                    csv_writer.writerow(tag2row(line))


if __name__ == "__main__":
    xml2tag('Posts.xml', 'AllTags.csv')
