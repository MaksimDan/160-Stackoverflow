import pandas as pd
import re
import progressbar


"""
File: map_tag_synonumns.py 
Objective: Map old tags to modified master tags.
"""


def remap_tag_synoymns(tag_syn_path, inpath):
    tag_syn_df = pd.read_csv(tag_syn_path)
    tag_map = {str(row['SourceTagName']).lower(): str(row['TargetTagName']).lower()
               for index, row in tag_syn_df.iterrows()}

    posts_df = pd.read_csv(inpath)
    posts_df['Tags'] = posts_df['Tags'].apply(lambda t: ' '.join(re.findall(r"\<(\w+)\>", str(t))))
    old_tag_list = posts_df.Tags.values
    new_tag_list = []

    bar = progressbar.ProgressBar()
    for tags in bar(old_tag_list):
        new_tag_list.append(' '.join([tag_map[tag] if tag.lower() in tag_map else tag for tag in tags.split()]))
    posts_df['Tags'] = new_tag_list
    posts_df.to_csv('PostsClean.csv')

	
if __name__ == "__main__":
	BASE_PATH = '../../160-Stackoverflow-Data/train_test/sqlite_subquery/'
	inpath = BASE_PATH + 'Posts_2012_Jan_May.csv'
	remap_tag_synoymns('../../160-Stackoverflow-Data/tags/TagSynonyms.csv', inpath)