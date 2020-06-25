from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pandas as pd


class ModelRanking(object):
    """
    Generates model ranking based on the ranking factor
    """

    def __init__(self, config, preds):
        self.config = config
        self.preds = preds
        self.cdrs_ml = pd.read_csv(os.path.join(self.config.data_dir, self.config.cdrs_ml_fname),
                                   sep=' ', dtype={'gvkey': str})
        self.cdrs_src = pd.read_csv(os.path.join(self.config.data_dir, self.config.cdrs_src_fname),
                                    sep=' ', dtype={'gvkey': str})

    def generate_ranking(self):
        self.preds = self.preds.merge(self.cdrs_ml[['date', 'gvkey', 'entval', 'tic', 'active']], how='left',
                                      left_on=['gvkey', 'date'], right_on=['gvkey', 'date'])
        self.preds = self.preds.merge(self.cdrs_src[['gvkey', 'sectorcd']].drop_duplicates(), how='left',
                                      left_on=['gvkey'], right_on=['gvkey'])

        self.preds['pred_var_entval'] = self.preds.apply(self.ey_pred_var, axis=1)
        self.preds['pred_entval'] = self.preds.apply(self.ey_pred, axis=1)

        self.preds = self.preds.sort_values(by=self.config.model_ranking_factor, ascending=False)

        self.preds.to_csv(self.config.model_ranking_fname, sep=' ', index=False)

    @staticmethod
    def ey_pred_var(row):
        try:
            factor = min(max(-1.0, row['pred_var_1'] / row['entval']), 1.0)
        except ZeroDivisionError:
            factor = -1000
        return factor

    @staticmethod
    def ey_pred(row):
        try:
            factor = min(max(-1.0, row['preds_1'] / row['entval']), 1.0)
        except ZeroDivisionError:
            factor = -1000
        return factor
