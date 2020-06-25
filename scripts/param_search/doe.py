from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import pickle
import copy
import platform
import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing as mp

from itertools import cycle, product
from param_search.opt_base_class import OptimizerBaseClass


class DOE(OptimizerBaseClass):
    """
    Design of Experiments
    """

    def __init__(self, search_config):
        self.search_config = search_config
        super().__init__(self.search_config)

        self.gpu_id_iterable = cycle(list(range(self.search_config.num_gpu)))

        self._id = 0

        self.best_mem = {'mem_id': 1,
                         'mem_config': None,
                         'mem_objective': np.inf}

        self.opt_hist = pd.DataFrame(columns=['id', 'gen', 'objective', 'valid_mse', 'valid_uq_loss'] + \
                                             sorted(self.tunable_vars) + \
                                             sorted(self.constant_vars))

        assert self.search_config.doe_file is not None

        self.doe = pd.read_csv(self.search_config.doe_file, sep=',')

        # create experiment dir
        if 'name' in self.doe.columns:
            self.exp_dirname = os.path.join(self.search_config.experiments_dir, self.doe['name'].iloc[0])
        else:
            self.exp_dirname = os.path.join(self.search_config.experiments_dir, self.search_config.name)

        if not os.path.isdir(self.exp_dirname):
            os.mkdir(self.exp_dirname)

        for col in self.doe.columns:
            if col not in list(self.search_config_dict.keys()):
                self.doe.drop(col, axis=1)

        self.num_gens = int(np.ceil(self.doe.shape[0] / self.NPE))
        self.pop_size = self.NPE

        self._cur_pop = None

    def _train_population(self, gen):
        """
        Runs training for configs in the current population and appends the results to opt history
        :param gen: generation number
        :return: population results type: tuple((member, objective))
        """

        for cur_mem in self._cur_pop:

            # update opt_hist dataframe
            self.opt_hist.loc[cur_mem.member_id, 'id'] = cur_mem.member_id
            self.opt_hist.loc[cur_mem.member_id, 'gen'] = gen

            for var in self.tunable_vars:
                self.opt_hist.loc[cur_mem.member_id, var] = cur_mem.__dict__['__configs'][var]

            for var in self.constant_vars:
                self.opt_hist.loc[cur_mem.member_id, var] = cur_mem.__dict__['__configs'][var]

        # execute iteration
        pool = mp.Pool(self.NPE)
        results = pool.map(self.execute_config, self._cur_pop)

        # update pop results for objective fxn
        for res in results:
            self.opt_hist.loc[res[-1].__dict__['__configs']['member_id'], 'objective'] = res[0]
            self.opt_hist.loc[res[-1].__dict__['__configs']['member_id'], 'valid_mse'] = res[0]
            self.opt_hist.loc[res[-1].__dict__['__configs']['member_id'], 'valid_uq_loss'] = res[1]

        return

    def execute_config(self, config):
        """
        Executes training for config
        :param config: config object
        :return:
        """

        config_dict = config.__dict__['__configs']

        # write config in model dir
        if not os.path.isdir(os.path.join(self.search_config.experiments_dir, config_dict['model_dir'])):
            os.mkdir(os.path.join(self.search_config.experiments_dir, config_dict['model_dir']))

        config_path = os.path.join(self.search_config.experiments_dir, config_dict['model_dir'], 'model.conf')

        with open(config_path, 'w') as f:
            for k, v in sorted(config_dict.items()):
                if v is not None:
                    if isinstance(v, (int, np.int64, np.int32,
                                      float, np.float64, np.float32,
                                      str,
                                      bool, np.bool, np.bool_)):
                        f.write('--{}       {} \n'.format(k, v))
                    elif isinstance(v, (list, tuple)):
                        f.write('--{}       '.format(k) + '-'.join('{}'.format(j) for j in v) + '\n')

        print(config_path)
        print("Train=%g" % config.train)
        os.system("CUDA_VISIBLE_DEVICES=%s python lfm_quant.py --config=%s" % (str(config.default_gpu)[-1],
                                                                                config_path))

        valid_loss, valid_uq_loss = self._read_results(config)
        return valid_loss, valid_uq_loss, config

    def _get_gen(self, gen_num):
        """
        returns next generation's population
        :param: generation number
        :return:
        """

        pop = self.doe.iloc[gen_num * self.NPE: (gen_num + 1) * self.NPE]
        pop_config = []

        for i in range(pop.shape[0]):
            mem_config = self._build_config(pop.iloc[i])
            pop_config.append(mem_config)

        return pop_config

    def _build_config(self, member):
        """
        Builds config object
        :param member: pandas series
        :return: config object
        """
        config = copy.deepcopy(self.search_config)
        if 'member_id' not in member.index:
            self._id += 1
            config.__dict__["__configs"]['member_id'] = self._id
            config.__dict__["__configs"]["model_dir"] = os.path.join(self.search_config.name,
                                                                     'opt_run',
                                                                     self.search_config.name + \
                                                                     '-' + str(self._id))

        # assign gpu
        config.__dict__['__configs']['default_gpu'] = '/gpu:' + str(next(self.gpu_id_iterable))

        # update variables that are constant
        for var in self.constant_vars:
            if var not in self.vars_keep_type:  # vars that need to be list type
                config.__dict__["__configs"][var] = self.search_config_dict[var][0]

        # update variables from the member pandas series
        for var in member.index:
            config.__dict__["__configs"][var] = member[var]

        for k, v in config.__dict__["__configs"].items():
            if k not in self.vars_keep_type:
                assert v is None or isinstance(v, (int, np.int64, np.int32,
                                                   float, np.float64, np.float32,
                                                   str,
                                                   bool, np.bool, np.bool_)), \
                    "All configuration parameters should have singular value. Received %g for param %s" % (v, k)

        return config

    @staticmethod
    def _read_results(config):
        """
        Returns the last epoch's result from training log
        :param config: case config; type: config object
        :return: validation loss
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
            else:
                # Read preds file
                df = pd.read_csv(os.path.join(config.experiments_dir,
                                              config.model_dir,
                                              'pred',
                                              config.preds_fname), sep=' ', dtype={'gvkey': str})
                valid_loss = 0
                valid_uq_loss = None

                for i in range(config.forecast_steps):
                    valid_loss += df['norm_squared_diff_' + str(i+1)].mean() * config.forecast_steps_weights[i]

        except FileNotFoundError:
            valid_loss = np.inf
            valid_uq_loss = np.inf

        return valid_loss, valid_uq_loss

    def run(self):
        """
        Executes doe with defined parameters
        :return:
        """

        random.seed(self.search_config.seed)
        print("Starting DOE ...")

        for i in range(self.num_gens):
            print("Gen %i" % (i + 1))
            self._cur_pop = self._get_gen(i)
            assert isinstance(self._cur_pop, (list, tuple))
            self._train_population(i + 1)
            self.opt_hist.to_csv(os.path.join(self.exp_dirname, 'opt_hist.csv'), sep=',', index=False)
