from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
import scipy.stats as st
from base_config import get_configs
from glob import glob

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


class PredictionsPostProcessing(object):
    """
    Post processing class for predictions.
    1. Aggregating ensemble results
    2. Generate lower bound forecasts for UQ models
    """

    def __init__(self, preds, config):
        self.preds = preds
        self.config = config

        self.df = preds[0]
        self.pred_fields = sorted([x for x in self.df.columns if 'preds' in x])
        self.var_fields = sorted([x for x in self.df.columns if 'variance' in x])
        self.tar_fields = sorted([x for x in self.df.columns if 'targets' in x])

        assert len(self.pred_fields) == len(self.var_fields) == len(self.tar_fields), "Number of preds and " \
                                                                                      "variance columns are " \
                                                                                      "different. " \
                                                                                      "They should be the same "

        for i in range(len(self.pred_fields)):
            assert self.pred_fields[i].split('_')[-1] == self.var_fields[i].split('_')[-1], "preds and variance are " \
                                                                                            "not matching "

    def _get_mean_df(self):
        """
        Aggregate predictions of models in the ensemble using mean
        :return: aggregated dataframe
        """

        for pred_field in self.pred_fields:
            pred_values = [x[pred_field].values for x in self.preds]
            mean_preds = np.mean(pred_values, axis=0)
            self.df.loc[:, pred_field] = mean_preds

        if self.config.UQ:
            for i, (pred_field, var_field) in enumerate(zip(self.pred_fields, self.var_fields)):
                var_values = [x[var_field].values for x in self.preds]
                noise_var = np.mean(var_values, axis=0)
                model_var = np.var(var_values, axis=0)
                self.df.loc[:, var_field] = noise_var + model_var
                # create column predictions/variance
                if 'norm' not in pred_field:
                    fcst_step = pred_field.split('_')[-1]
                    self.df.loc[:, 'pred_var_' + fcst_step] = self.df.loc[:, pred_field] / \
                                                              np.sqrt(self.df.loc[:, var_field])

        for step in range(self.config.forecast_steps):
            self.df['norm_squared_diff_' + str(step + 1)] = (self.df['norm_targets_' + str(step + 1)] -
                                                             self.df['norm_preds_' + str(step + 1)]).apply(np.square)

            self.df['abs_err_' + str(step + 1)] = (self.df['targets_' + str(step + 1)] -
                                                   self.df['preds_' + str(step + 1)]).abs()

            self.df['fcst_err_' + str(step + 1)] = self.df['abs_err_' + str(step + 1)] / self.df[
                'targets_' + str(step + 1)].abs()

            self.df['unscaled_squared_err_' + str(step + 1)] = (self.df['abs_err_' + str(step + 1)] /
                                                               self.df['seq_norm']).apply(np.square)

        return self.df

    def aggregate_point_estimate(self):
        df = self._get_mean_df()
        for i in range(self.config.forecast_steps):
            print("Scaled MSE for step %i: %1.4f" % (i + 1, df['norm_squared_diff_' + str(i + 1)].mean()))
        return df

    def aggregate_uq_estimate(self, certainty_level=None):
        """
        Generates lower bound dataframes using UQ predictions
        :param certainty_level: certainty level to calculate the lower bounds on. list(float) between 0 - 100.
        Example: certainty_level=50 means LB calculated using the prediction intervals that contain 50% of the
        true values
        :return: dataframes for each level
        """
        if certainty_level is None:
            certainty_level = [0.5]
        assert isinstance(certainty_level, (list, tuple))

        df = self._get_mean_df()

        # check if all the targets are nan i.e predictions are for dates for which targets are not available
        # for example today.
        targets_1 = df[self.tar_fields[0]]
        if targets_1.isnull().sum() == targets_1.shape[0]:
            # add empty columns for lower bond vars
            for i in range(len(self.pred_fields)):
                for c_level in certainty_level:
                    level_name = str(int(c_level * 100))
                    df[self.pred_fields[i] + "_" + level_name] = np.nan
            return df

        # p-value to z-score
        p_val = np.linspace(0.0001, 0.999999, 100)
        z_score = st.norm.ppf(p_val)
        freq = np.array([self._get_frequency(z, df) for z in z_score])

        for i in range(len(self.pred_fields)):
            iso_reg = IsotonicRegression().fit(freq[:, i], z_score)
            for c_level in certainty_level:
                calib_z_score = iso_reg.predict([c_level])
                level_name = str(int(c_level * 100))

                df.loc[:, self.pred_fields[i] + "_" + level_name] = \
                    df[self.pred_fields[i]].values - calib_z_score * np.sqrt(df[self.var_fields[i]].values)

        for i in range(self.config.forecast_steps):
            print("Scaled MSE for step %i: %1.4f" % (i + 1, df['norm_squared_diff_' + str(i + 1)].mean()))
        return df

    def _get_frequency(self, z_score, df):
        """
        returns the percentage of true values lying between the interval generated by z_score
        :param z_score: z_score
        :param df: prediction dataframe
        :return:
        """
        freq = []
        for i in range(len(self.pred_fields)):
            lower_bound = df[self.pred_fields[i]].values - z_score * np.sqrt(df[self.var_fields[i]].values)
            upper_bound = df[self.pred_fields[i]].values + z_score * np.sqrt(df[self.var_fields[i]].values)
            lower = np.less_equal(lower_bound, df[self.tar_fields[i]].values)
            upper = np.less_equal(df[self.tar_fields[i]].values, upper_bound)
            inside_interval = np.logical_and(lower, upper)
            freq_perc = 1. * np.sum(inside_interval) / \
                        (len(inside_interval) - np.sum(np.isnan(df[self.tar_fields[i]].values)))
            freq.append(freq_perc)

        return freq


if __name__ == '__main__':
    config = get_configs()
    config.UQ = False
    config.train = False
    config.forecast_steps = 1

    preds_paths = glob('../../dev_test/test-preds/*')

    preds = [pd.read_csv(path, sep=' ', dtype={'gvkey': str}) for path in preds_paths]
    post_proc = PredictionsPostProcessing(preds, config)

    agg_preds = post_proc.aggregate_uq_estimate([0.5])

    print(agg_preds.head())
