import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from scipy import stats
from sklearn.metrics import roc_curve, auc
plt.style.use('ggplot')


class DataUtilities:
    # https://stats.stackexchange.com/questions/51248/how-can-i-find-the-standard-deviation-in-categorical-distribution
    @staticmethod
    def entropy(cata_vector):
        """
        objective: compute the measure of entropy of a catagorical vector
        :param cata_vector: list(factor)
        :return: float - shermons entropy level
        """
        px = np.matrix(stats.itemfreq(cata_vector))[:, 1] / len(cata_vector)
        lpx = np.log2(px)
        ent = -np.sum(px.T * lpx)
        return ent

    @staticmethod
    def build_residual_dataframe(observed_ranks):
        """
        objective: convert residuals dictionary from engine to a dataframe
        :param observed_ranks: dict - observed ranks by type of activity
        :return: equivalent pd.Dataframe
        """
        df = pd.DataFrame(DataUtilities.flatten_residual_dictionary(observed_ranks))
        _max_rank = max(df['rank'].values)
        df['rank'] = df['rank'].apply(lambda x: x / _max_rank)
        return df

    @staticmethod
    def flatten_residual_dictionary(r_dict):
        """
        objective: flatten json residual structure
        :param r_dict: dict - residual dictionary
        :return: list of flattened dictionary mappings
        """
        flatted_d = []
        for question_i, activities in r_dict.items():
            for activity, index_list in activities.items():
                for rank in index_list:
                    flatted_d.append({'question_number': question_i, 'rank': rank, 'activity': activity})
        return flatted_d

    @staticmethod
    def flatten_full_residual_dictionary(r_dict):
        """
        objective: flatten json residual structure
        :param r_dict: dict - residual dictionary
        :return: list of flattened dictionary mappings
        """
        flatted_d = []
        for question_i, activities in r_dict.items():
            for activity, data_types in activities.items():
                for u_id, rank, score in zip(data_types['userid'], data_types['rank'], data_types['score']):
                    flatted_d.append({'q_num': question_i, 'activity': activity,
                                      'userid': u_id, 'rank': rank, 'score': score})
        return flatted_d

    @staticmethod
    def build_threshold_dataframe(raw_residuals):
        """
        objective: builds a dataframe used to compute thresholds
        :param raw_residuals: dict - residual dictionary
        :return: pd.Dataframe
        """
        df = DataUtilities.build_residual_dataframe(raw_residuals)

        def find_nearest(array, value):
            # search sorted finds the index where an element would be inserted to maintain order in the array
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
    def min_max_scale(ar):
        return ar/max(ar)


class ResidualPlots:
    col_list = ["blue", "cyan", "green", "red"]
    col_list_palette = sns.xkcd_palette(col_list)

    @staticmethod
    def plot_residual_matrix(raw_residuals, save_path):
        """
        objective: visualizes the main summary plot
        :param raw_residuals: dict - residuals from engine
        :param save_path: string - path to save plot
        :return:
        """
        df = DataUtilities.build_residual_dataframe(raw_residuals)
        sns.set_palette(ResidualPlots.col_list_palette)

        g = sns.lmplot('rank', 'question_number', data=df, hue='activity',
                       fit_reg=False, palette=ResidualPlots.col_list_palette, markers='s',
                       scatter_kws={"s": 10})

        # colors for vertical lines
        g.set(xticks=[])
        g.set(yticks=[])
        ax = plt.gca()
        ax.invert_yaxis()

        col_dict = {'i_answer': 'blue', 'i_comment': 'cyan',
                    'i_editor': 'green', 'i_favorite': 'red'}
        for act, sub_df in df.groupby('activity'):
            plt.axvline(x=sub_df['rank'].median(), ymax=0.96, color=col_dict[act])

        plt.gcf().suptitle("Residual Matrix")
        plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_rank_error_by_threshold(raw_residuals, save_path):
        """
        objective: visualize performance of the model by threshold
        :param raw_residuals: dict - residual dictionary
        :param save_path: string - path to save plot
        :return:
        """
        threshold_error_by_activity_df = DataUtilities.build_threshold_dataframe(raw_residuals)
        sns.lmplot('t', 'capture_accuracy', data=threshold_error_by_activity_df, hue='activity', fit_reg=False,
                   palette=ResidualPlots.col_list_palette, markers='.')
        plt.gcf().suptitle('Threshold Accuracy by User Activity')
        plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_rank_error_distributions(raw_residuals, save_path):
        """
        objective: plots the distribution of rank by activity type
        :param raw_residuals: dict - residuals dictionary
        :param save_path: string - path to save plot
        :return:
        """
        df = DataUtilities.build_residual_dataframe(raw_residuals)
        activity_groups = df.groupby('activity')
        n_error_types = len(activity_groups)
        fig, axs = plt.subplots(1, n_error_types, figsize=(15, 6))
        axs = axs.ravel()

        for i, (activity, sub_df) in enumerate(activity_groups):
            if len(sub_df['rank'].values) <= 1:
                logging.debug(f'Cannot plot error distribution for activity {activity}, not enough data.')
            else:
                sns.distplot(sub_df['rank'].values, ax=axs[i]).set_title(activity)

        fig.suptitle('Rank Distribution by Activity Level', fontsize=14)
        plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_loss_function_error_distribution(loss_errors, save_path):
        """
        objective: plots the distribution of loss function errors per question
        :param loss_errors: list(float) - loss function errors
        :param save_path: string - path to save plot
        :return:
        """
        plt.hist(loss_errors, 20, normed=1)
        plt.title('Distribution of all Loss Function Errors')
        plt.savefig(save_path)
        plt.xlabel('Error')
        plt.ylabel('Frequency (normalized)')
        plt.show()

    @staticmethod
    def plot_entropy_per_rank(rank_matrix, save_path):
        """
        objective: plots shannons entropy by column rank
        :param rank_matrix: NxM matrix where rows are questions, and columns are users
        :param save_path: string - path to save plot
        :return:
        """
        rank = np.arange(0, rank_matrix.shape[1])
        entropy_ar = np.array([DataUtilities.entropy(rank_matrix[:, j]) for j in rank], dtype=np.float)
        sns.lmplot('rank', 'entropy', data=pd.DataFrame({'rank': rank, 'entropy': entropy_ar}),
                   fit_reg=False, markers='.')
        # alternative, but less clear: sns.kdeplot(rank, entropy_ar, shade=True)
        plt.title('Recommender System Entropy by Rank')
        plt.xlabel('User Rank')
        plt.ylabel('Entropy')
        plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_roc_curve_for_all_activities(score_matrix, label_matrix, save_path):
        """
        objective: plots receiver operating characteristic curve on classification
        :param score_matrix: NxM matrix where rows ar questions, and columns are users
        :param label_matrix: NxM matrix where rows are questions, and columns are indicative
                             if there was an activity by user_j
        :param save_path: string - path to save plot
        :return:
        """
        fpr, tpr, _ = roc_curve(label_matrix.ravel(), score_matrix.ravel())
        roc_auc = auc(fpr, tpr)
        # np.savetxt("score_matrix.csv", score_matrix, delimiter=",")
        # np.savetxt("label_matrix.csv", label_matrix, delimiter=",")

        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve by Score Threshold')
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.show()

    def plot_weight_vs_error(self, weight_error_df, split_point, save_path):
        """
        objective: plots the change of error based on weight value of feature
        :param weight_error_df: pd.Dataframe
        :param split_point: int - number of rows in df until next feature
        :param save_path: string - path to save plot
        :return:
        """
        individual_df = {}
        row_index = np.arange(0, len(weight_error_df) + split_point, split_point)
        feature_order = list(filter(lambda x: x not in ['loss_function_error', 'rank_error'], list(weight_error_df)))

        for i in range(1, len(row_index)):
            row_start, row_end = row_index[i - 1], row_index[i]
            feature_name = feature_order[i - 1]
            individual_df[feature_name] = \
                weight_error_df[[feature_name, 'rank_error', 'loss_function_error']].iloc[row_start:row_end, :]

        fig, axs = plt.subplots(4, 2, figsize=(6, 8))
        fig.tight_layout()
        axs = axs.ravel()

        for i, (weight_type, df) in enumerate(individual_df.items()):
            df['rank_error'] = DataUtilities.min_max_scale(np.array(df['rank_error']))
            df['loss_function_error'] = DataUtilities.min_max_scale(np.array(df['loss_function_error']))
            df.plot.line(x=weight_type, y='rank_error', ax=axs[i])
            df.plot.line(x=weight_type, y='loss_function_error', ax=axs[i])
            axs[i].legend_.remove()

        plt.figlegend(labels=('Rank Sum', 'Error'), loc='lower center', labelspacing=0)

        plt.subplots_adjust(top=.9, hspace=.5, wspace=.25, bottom=.15)
        fig.suptitle('Error by Feature Weight', fontsize=18)
        plt.savefig(save_path)
        plt.show()

