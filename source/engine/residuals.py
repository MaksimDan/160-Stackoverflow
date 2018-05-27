import json
from collections import defaultdict

import numpy as np

from visuals import ResidualPlots
from math import log10

def decay(x):
    return log10(1) - 2*log10(x)

class Residuals:
    def __init__(self, X, y):
        self.X, self.y = X, y
        self.error_per_question = defaultdict(list)
        self.raw_residuals_per_question = defaultdict(lambda: defaultdict(lambda: list()))
        self.user_rank = []

    def compute_and_store_residuals(self, score_matrix, y_index):
        observed = self.y.iloc[y_index]
        #TODO: Get the variability of user rank here
        predicted = {int(score_matrix[i, 0]): {'index': i, 'score': score_matrix[i, -1]} for i in
                     range(score_matrix.shape[0])}
        self.user_rank.append(list(predicted))

        
        index_answer = [predicted[user]['index'] for user in eval(observed.answerers)]
        index_comment = [predicted[user]['index'] for user in eval(observed.commenters)]
        # index_editor = [predicted[user]['index'] for user in eval(observed.editers)]
        # index_favorite = [predicted[user]['index'] for user in eval(observed.favorite)]

        self.error_per_question['answer'].append(sum(index_answer))
        self.error_per_question['comment'].append(sum(index_comment))
        # self.error_per_question['edit'].append(sum(index_editor))
        # self.error_per_question['favorite'].append(sum(index_favorite))

        self.raw_residuals_per_question[y_index]['i_answer'] = index_answer
        self.raw_residuals_per_question[y_index]['i_comment'] = index_comment
        # self.raw_residuals_per_question[y_index]['i_editor'] = index_editor
        # self.raw_residuals_per_question[y_index]['i_favorite'] = index_favorite

    def get_total_error(self):
        d = self.flatted_errors()
        return d['answer'] + d['comment']

    def get_summarize_statistics(self, n_total_users):
        stats = 'Residual Summary Statistics\n\n'

        stats += 'Error per User Activity:\n'
        stats += Residuals.pprint_dict(self.flatted_errors())

        stats += '\nAverage Percent Error per User Activity:\n'
        stats += Residuals.pprint_dict(self.get_average_percent_rank_error_per_question(n_total_users))

        stats += '\nThreshold Accuracy per User Activity:\n'
        stats += Residuals.pprint_dict(self.summarize_error_by_threshold())

        n_questions = len(self.raw_residuals_per_question)
        stats += '\nAverage Rank Error Per Question: '
        stats += str(self.get_total_error() / n_questions) + '\n'

        stats += 'Total Error: '
        stats += str(self.get_total_error()) + '\n'

        return stats

    def flatted_errors(self):
        flatted_errors = defaultdict(int)
        for activity, ranks in self.error_per_question.items():
            flatted_errors[activity] = sum(ranks)
        return flatted_errors

    def get_average_percent_rank_error_per_question(self, n_total_users):
        d = self.flatted_errors()
        n_questions = len(self.raw_residuals_per_question)
        for activity, sum_rank_error in d.items():
            d[activity] = sum_rank_error / n_questions / n_total_users
        return d

    def summarize_error_by_threshold(self):
        r_plot = ResidualPlots()
        t_df = r_plot.build_threshold_dataframe(self.raw_residuals_per_question)

        threshold_summary = {}
        summary_marker = np.arange(0, 1 + .1, .1)
        for activity, sub_df in t_df.groupby('activity'):
            # normalize the positions where the thresholds are, in order to observe relative position
            _max_position = max(sub_df['capture_accuracy'].values)
            sub_df['capture_accuracy'] = sub_df['capture_accuracy'].apply(lambda x: x / _max_position)
            basic_increment_index = np.where(np.isin(np.array(sub_df['t']), summary_marker))
            summary_threshold = np.array(sub_df['capture_accuracy'])[basic_increment_index]
            threshold_summary[activity] = {f'{round(t*100, 1)}%': t_error
                                           for t, t_error in zip(summary_marker, summary_threshold)}
        return threshold_summary

    @staticmethod
    def pprint_dict(d):
        return json.dumps(d, indent=4)
