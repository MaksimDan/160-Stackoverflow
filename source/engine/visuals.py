import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class ResidualPlots:
    col_list = ["blue", "cyan", "purple", "yellow"]
    col_list_palette = sns.xkcd_palette(col_list)

    @staticmethod
    def _build_residual_dataframe(observed_ranks):
        df = pd.DataFrame(ResidualPlots.__flatten_residual_dictionary(observed_ranks))
        _max_rank = max(df['rank'].values)
        df['rank'] = df['rank'].apply(lambda x: x / _max_rank)
        return df

    @staticmethod
    def __flatten_residual_dictionary(r_dict):
        flatted_d = []
        for question_i, activities in r_dict.items():
            for activity, index_list in activities.items():
                for rank in index_list:
                    flatted_d.append({'question_number': question_i, 'rank': rank, 'activity': activity})
        return flatted_d

    @staticmethod
    def plot_residual_matrix(raw_residuals, save_path):
        df = ResidualPlots._build_residual_dataframe(raw_residuals)
        sns.set_palette(ResidualPlots.col_list_palette)

        g = sns.lmplot('rank', 'question_number', data=df, hue='activity',
                       fit_reg=False, palette=ResidualPlots.col_list_palette, markers='s',
                       scatter_kws={"s": 10})

        # colors for vertical lines
        keys = ['i_answer', 'i_comment', 'i_edit', 'i_favorite']
        dic = dict(zip(keys, ResidualPlots.col_list))

        g.set(xticks=[])
        g.set(yticks=[])
        ax = plt.gca()
        ax.invert_yaxis()

        avg_error = df.groupby('activity')['rank'].mean()
        for err in avg_error:
            plt.axvline(x=err, ymax=0.96, color=dic.get(avg_error[avg_error == err].index[0]))

        plt.gcf().suptitle("Residual Matrix")
        plt.savefig(save_path)
        plt.show()

    @staticmethod
    def build_threshold_dataframe(raw_residuals):
        df = ResidualPlots._build_residual_dataframe(raw_residuals)

        def find_nearest(array, value):
            # searchsorted finds the index where an element would be inserted to maintain order in the array
            idx = np.searchsorted(array, value, side='right')
            if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
                return idx - 1
            else:
                return idx

        threshold_error_by_activity = []
        for activity, sub_df in df.groupby('activity'):
            sorted_ranks = np.array(sorted(sub_df['rank'].values))
            for threshold in np.arange(0, 1 + .01, .01):
                threshold_error_by_activity.append({'activity': activity,
                                                    't': threshold,
                                                    'capture_accuracy': find_nearest(sorted_ranks, threshold) / len(sorted_ranks)})
        return pd.DataFrame(threshold_error_by_activity)

    @staticmethod
    def plot_error_by_threshold(raw_residuals, save_path):
        threshold_error_by_activity_df = ResidualPlots.build_threshold_dataframe(raw_residuals)
        sns.lmplot('t', 'capture_accuracy', data=threshold_error_by_activity_df, hue='activity', fit_reg=False,
                   palette=ResidualPlots.col_list_palette, markers='.')
        plt.gcf().suptitle('Threshold Accuracy by User Activity')
        plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_error_distributions(raw_residuals, save_path):
        df = ResidualPlots._build_residual_dataframe(raw_residuals)
        activity_groups = df.groupby('activity')
        n_error_types = len(activity_groups)
        fig, axs = plt.subplots(1, n_error_types, figsize=(15, 6))
        axs = axs.ravel()

        for i, (activity, sub_df) in enumerate(activity_groups):
            sns.distplot(sub_df['rank'].values, ax=axs[i]).set_title(activity)

        fig.suptitle('Rank Distribution by Activity Level', fontsize=14)
        plt.savefig(save_path)
        plt.show()
