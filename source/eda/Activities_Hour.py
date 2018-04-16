
import pandas as pd

Comments = pd.read_csv('/Users/seungmi/Dropbox/160-Stackoverflow-Data/100000_rows_[168_MB]/Comments.csv')
Posts = pd.read_csv('/Users/seungmi/Dropbox/160-Stackoverflow-Data/100000_rows_[168_MB]/Posts.csv')
Users = pd.read_csv('/Users/seungmi/Dropbox/160-Stackoverflow-Data/100000_rows_[168_MB]/Users.csv')

def user_avail(user_id):
    user_questions = Posts.loc[(Posts.OwnerUserId == user_id)&(Posts.PostTypeId == 1), ['CreationDate']]
    user_answers = Posts.loc[(Posts.OwnerUserId == user_id)&(Posts.PostTypeId == 2), ['CreationDate']]
    user_comments = Comments.loc[Comments.UserId == user_id, ['CreationDate']]
    user_activities = user_questions.append(user_answers).append(user_comments)
    user_hours = pd.DataFrame({'Hours':user_activities.CreationDate.str[11:13]})
    grouped = user_hours['Hours'].groupby(user_hours['Hours'])
    activities_hours = grouped.count()
    # activities_prop = activities_hours/sum(activities_hours)
    return activities_hours # activities_prop
