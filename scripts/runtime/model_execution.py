from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import platform
import random
import copy
import multiprocessing as mp
import tensorflow as tf
from itertools import cycle
import numpy as np
import pandas as pd
import pathlib

from utils.post_processing import PredictionsPostProcessing
from utils.model_ranking import ModelRanking


class ModelExecution(object):

    def __init__(self, config):
        self.config = config

    def __call__(self):
        if self.config.training_type == 'iterative':
            raise NotImplementedError
        else:
            self.fixed_dates_execution()

    @staticmethod
    def read_results(config):
        """
        Returns validation metrics if training; and predictions if inference
        :param config: case config; type: config object
        :return: validation_loss, validat
        """

        try:
            if config.train:
                # Read the train log
                df = pd.read_csv(os.path.join(config.experiments_dir,
                                              config.model_dir,
                                              'train_log',
                                              config.name + '-train-logs-epoch.csv'), sep=',')

                if not config.UQ:
                    df = df.sort_values(by='valid_mse').reset_index()
                    valid_loss = df.iloc[0]['valid_mse']
                    valid_uq_loss = None
                else:
                    df = df.sort_values(by='valid_uq_loss').reset_index()
                    valid_loss = df.iloc[0]['valid_mse']
                    valid_uq_loss = df.iloc[0]['valid_uq_loss']

                preds = None

            else:
                # Read preds file
                pred_path = os.path.join(config.experiments_dir,
                                         config.model_dir,
                                         'pred',
                                         config.preds_fname)
                df = pd.read_csv(pred_path, sep=' ', dtype={'gvkey': str})
                valid_loss = 0
                valid_uq_loss = None

                for i in range(config.forecast_steps):
                    valid_loss += df['norm_squared_diff_' + str(i + 1)].mean() * config.forecast_steps_weights[i]

                preds = df

        except FileNotFoundError:
            print("Output file not found")
            valid_loss = np.inf
            valid_uq_loss = np.inf
            preds = None

        return valid_loss, valid_uq_loss, preds

    @staticmethod
    def single_execution(config):
        """
        Executes training/prediction for the config as a single individual process
        :param config: config object
        :return:
        """

        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.default_gpu)[-1]

        from data_processing import CDRSInferenceData, Dataset
        from train import Train
        from predict import Predict

        if config.cdrs_inference:
            dataset = CDRSInferenceData(config)
        else:
            dataset = Dataset(config)

        # training
        if config.train:
            print("Training")
            train = Train(config, dataset)
            train.train()

        # prediction
        else:
            print("Prediction")
            pred = Predict(config, dataset)
            pred.predict()

        return

    def execute_config(self, config):
        """
        Executes training/ prediction of the config file as a part of an ensemble
        :param config:
        :return:
        """

        # Individual config files are run as a single process with num_procs=1
        config.__dict__["__configs"]['num_procs'] = 1
        config.__dict__["__configs"]['NPE'] = 1

        config_dict = config.__dict__['__configs']

        # write config in model dir
        if not os.path.isdir(os.path.join(config.experiments_dir, config_dict['model_dir'])):
            os.mkdir(os.path.join(config.experiments_dir, config_dict['model_dir']))

        config_path = os.path.join(config.experiments_dir,
                                   config_dict['model_dir'],
                                   'model-' + str(config.member_id) + '.conf')

        with open(config_path, 'w') as f:
            for k, v in sorted(config_dict.items()):
                if v is not None:
                    if isinstance(v, (int, float, str, bool)):
                        f.write('--{}       {} \n'.format(k, v))
                    elif isinstance(v, (list, tuple)):
                        f.write('--{}       '.format(k) + '-'.join('{}'.format(j) for j in v) + '\n')

        os.system("CUDA_VISIBLE_DEVICES=%s lfm_quant.py --config=%s" % (str(config.default_gpu)[-1],
                                                                                config_path))
        valid_loss, valid_loss_fcst, preds = self.read_results(config)

        return valid_loss, valid_loss_fcst, preds

    def fixed_dates_execution(self):
        if self.config.UQ:  # uq range estimate
            self.uq_estimate_execution()
        else:  # point estimates
            self.point_estimate_execution()

    def point_estimate_execution(self):

        if self.config.num_procs == 1:
            self.single_execution(self.config)

        else:  # ensemble prediction
            assert self.config.num_procs > 1

            configs_list = []
            gpu_id_iterable = cycle(list(range(self.config.num_gpu)))

            for case_id in range(self.config.num_procs):
                base_config = copy.deepcopy(self.config)

                # create experiment_dir inside experiments
                assert os.path.isdir(base_config.experiments_dir), \
                    "Experiments dir  %s does not exist" % base_config.experiments_dir

                experiment_dir_path = os.path.join(base_config.experiments_dir, base_config.model_dir)

                if not os.path.isdir(experiment_dir_path):
                    print("Creating experiment dir %s" % experiment_dir_path)
                    pathlib.Path(experiment_dir_path).mkdir(parents=True, exist_ok=True)

                # update case specific information
                base_config.__dict__['__configs']['name'] = base_config.name
                base_config.__dict__['__configs']['member_id'] = case_id
                # base_config.__dict__['__configs']['preds_fname'] = 'preds-' + str(case_id + 1) + '.dat'
                base_config.__dict__['__configs']['model_dir'] = os.path.join(experiment_dir_path,
                                                                              base_config.name + '-' + str(case_id))
                base_config.__dict__['__configs']['default_gpu'] = '/gpu:' + str(next(gpu_id_iterable))

                # first case is run with the seed provided in the config file
                if case_id > 0:
                    base_config.__dict__['__configs']['seed'] = random.randint(0, 1000)

                configs_list.append(base_config)

            pool = mp.Pool(self.config.NPE)
            results = pool.map(self.execute_config, configs_list)

            if not self.config.train:
                preds = [res[2] for res in results]
                post_proc = PredictionsPostProcessing(preds, self.config)
                agg_preds = post_proc.aggregate_point_estimate()
                agg_preds.to_csv(os.path.join(experiment_dir_path, 'ensemble_preds.dat'), sep=' ', index=False)

    def uq_estimate_execution(self):
        if self.config.train or self.config.num_procs == 1:
            self.single_execution(self.config)

        else:  # ensemble prediction
            assert self.config.num_procs > 1, "UQ inference requires more than 1 num_procs"

            configs_list = []
            gpu_id_iterable = cycle(list(range(self.config.num_gpu)))

            for case_id in range(self.config.num_procs):
                base_config = copy.deepcopy(self.config)

                # create experiment_dir inside experiments
                assert os.path.isdir(base_config.experiments_dir), \
                    "Experiments dir  %s does not exist" % base_config.experiments_dir

                experiment_dir_path = os.path.join(base_config.experiments_dir, base_config.model_dir)

                assert (os.path.isdir(experiment_dir_path)), 'Model dir %s does not exist to read the model" \
                                                            '"params from" % experiment_dir_path

                # update case specific information
                base_config.__dict__['__configs']['member_id'] = case_id + 1
                base_config.__dict__['__configs']['preds_fname'] = 'preds-' + str(case_id + 1) + '.dat'
                base_config.__dict__['__configs']['default_gpu'] = '/gpu:' + str(next(gpu_id_iterable))

                # first case is run with the seed provided in the config file
                if case_id > 0:
                    base_config.__dict__['__configs']['seed'] = random.randint(0, 1000)

                configs_list.append(base_config)

            pool = mp.Pool(self.config.NPE)
            results = pool.map(self.execute_config, configs_list)

            if not self.config.train:
                preds = [res[2] for res in results]
                post_proc = PredictionsPostProcessing(preds, self.config)
                agg_preds = post_proc.aggregate_uq_estimate()
                agg_preds.to_csv(os.path.join(experiment_dir_path, 'ensemble_preds.dat'), sep=' ', index=False)

                if self.config.cdrs_inference:
                    print("Generating ranking data")
                    model_ranking = ModelRanking(self.config, agg_preds)
                    model_ranking.generate_ranking()
