import pandas as pd
import re
import progressbar


def remap_tag_synoymns(tag_syn_path, post_inpaths):
    tag_syn_df = pd.read_csv(tag_syn_path)
    # for index, row in tag_syn_df.iterrows():
    #     print(row['SourceTagName'].lower(), row['TargetTagName'].lower())

    tag_map = {str(row['SourceTagName']).lower(): str(row['TargetTagName']).lower()
               for index, row in tag_syn_df.iterrows()}

    for i, path in enumerate(post_inpaths):
        posts_df = pd.read_csv(path)
        posts_df['Tags'] = posts_df['Tags'].apply(lambda t: ' '.join(re.findall(r"\<(\w+)\>", str(t))))
        old_tag_list = posts_df.Tags.values
        new_tag_list = []

        bar = progressbar.ProgressBar()
        for tags in bar(old_tag_list):
            new_tag_list.append(' '.join([tag_map[tag] if tag.lower() in tag_map else tag for tag in tags.split()]))
        posts_df['Tags'] = new_tag_list
        posts_df.to_csv(f'PostsClean{i}.csv')


if __name__ == '__main__':
    inpaths = ['../../160-Stackoverflow-Data/2500_rows/Posts.csv',
               '../../160-Stackoverflow-Data/100000_rows/Posts.csv',
               '../../160-Stackoverflow-Data/300000_rows/Posts.csv']
    remap_tag_synoymns('../../160-Stackoverflow-Data/tags/TagSynonyms.csv', inpaths)
