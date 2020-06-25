from __future__ import absolute_import, division, print_function, unicode_literals

import os
from datetime import datetime
from pathlib import Path
from cdrs.core.dq_ml_data import DqMlData


class GenerateCDRSData(object):

    def __init__(self, config):
        self.config = config

    def __call__(self):
        if self.config.cdrs_inference_date is None:
            exec_date = datetime.now().strftime('%Y-%m-%d')
        else:
            exec_date = datetime.strptime(self.config.cdrs_inference_date, "%Y-%m-%d")

        # create datasets dir
        if not os.path.isdir(self.config.data_dir):
            print("Creating data dir %s" % self.config.data_dir)
            Path(self.config.data_dir).mkdir()

        DqMlData(exec_date=exec_date,
                 seq_len=self.config.max_unrollings,
                 datasets_dir=self.config.data_dir,
                 src_file_name=self.config.cdrs_src_fname,
                 ml_file_name=self.config.cdrs_ml_fname)
