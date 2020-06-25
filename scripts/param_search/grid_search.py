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


class GridSearch(OptimizerBaseClass):
    """
    Genetic Algorithm Optimization
    """

    def __init__(self, search_config):
        self.search_config = search_config
        super().__init__(self.search_config)

        self.gpu_id_iterable = cycle(list(range(self.search_config.num_gpu)))

        self._id = 0

        self.best_mem = {'mem_id': 1,
                         'mem_config': None,
                         'mem_objective': np.inf}

        self.opt_hist = pd.DataFrame(columns=['id', 'gen', 'objective', 'valid_mse', 'valid_mse_fcst'] + \
                                             sorted(self.tunable_vars) + \
                                             sorted(self.constant_vars))

        # create experiment dir
        self.exp_dirname = os.path.join(self.search_config.experiments_dir, self.search_config.name)
        if not os.path.isdir(self.exp_dirname):
            os.mkdir(self.exp_dirname)

        # create opt_run dir
        self.opt_run_dir = os.path.join(self.exp_dirname, 'opt_run')
        if not os.path.isdir(self.opt_run_dir):
            os.mkdir(self.opt_run_dir)

        # Create all combinations
        tunable_vars_val = []
        for k in self.tunable_vars:
            tunable_vars_val.append(self.search_config_dict[k])

        self.doe = list(product(*tunable_vars_val))
        print("Tunable Params")
        print(self.tunable_vars)

        self.num_gens = len(self.doe) // self.NPE
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
            self.opt_hist.loc[res[-1].__dict__['__configs']['member_id'], 'valid_mse_fcst'] = res[1]

        return

    def execute_config(self, config):
        """
        Executes training for config
        :param config: config object
        :return:
        """

        config.__dict__["__configs"]['train'] = True
        # Individual config files are run as a single process with num_procs=1
        config.__dict__["__configs"]['num_procs'] = 1
        config.__dict__["__configs"]['NPE'] = 1

        config_dict = config.__dict__['__configs']

        # write config in model dir
        if not os.path.isdir(os.path.join(self.search_config.experiments_dir, config_dict['model_dir'])):
            os.mkdir(os.path.join(self.search_config.experiments_dir, config_dict['model_dir']))

        config_path = os.path.join(self.search_config.experiments_dir, config_dict['model_dir'], 'model.conf')

        with open(config_path, 'w') as f:
            for k, v in sorted(config_dict.items()):
                if v is not None:
                    if isinstance(v, (int, float, str, bool)):
                        f.write('--{}       {} \n'.format(k, v))
                    elif isinstance(v, (list, tuple)):
                        f.write('--{}       '.format(k) + '-'.join('{}'.format(j) for j in v) + '\n')

        os.system("CUDA_VISIBLE_DEVICES=%s python deep_quant.py --config=%s" % (str(config.default_gpu)[-1],
                                                                                config_path))

        valid_loss, valid_loss_fcst = self._read_results(config)

        return valid_loss, valid_loss_fcst, config

    def _get_gen(self, gen_num):
        """
        returns next generation's population
        :param: generation number
        :return:
        """

        pop = self.doe[gen_num * self.NPE: (gen_num + 1) * self.NPE]
        pop_config = []

        for mem in pop:
            mem_config = self._build_config(mem)
            pop_config.append(mem_config)

        return pop_config

    def _build_config(self, member):
        """
        Builds config object
        :param member: tuple of values of tunable parameters. If tunable_vars = [a,b,c], member=(1,2,3)
        then a=1, b=2, c=3
        :return: config object
        """

        assert len(member) == len(self.tunable_vars)

        config = copy.deepcopy(self.search_config)
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

        # assign tunable variables from member
        for i, var in enumerate(self.tunable_vars):
            if var not in self.vars_keep_type:  # vars that need to be list type
                config.__dict__["__configs"][var] = member[i]

        for k, v in config.__dict__["__configs"].items():
            if k not in self.vars_keep_type:
                assert v is None or isinstance(v, (int, float, str, bool)), \
                    "All configuration parameters should have singular value. Received %g for param %s" % (v, k)

        return config

    @staticmethod
    def _read_results(config):
        """
        Returns the last epoch's result from training log
        :param config: case config; type: config object
        :return: validation loss
        """

        # TODO: Add uq loss as done in doe.py

        if config.UQ:
            raise NotImplementedError
        else:
            try:
                # Read the train log
                df = pd.read_csv(os.path.join(config.experiments_dir,
                                              config.model_dir,
                                              'train_log',
                                              config.name + '-train-logs-epoch.csv'), sep=',')

                df = df.sort_values(by='valid_mse').reset_index()
                valid_loss = df.iloc[0]['valid_mse']
                valid_loss_fcst = df.iloc[0]['valid_mse_fcst']

            except FileNotFoundError:
                valid_loss = np.inf
                valid_loss_fcst = np.inf

        return valid_loss_fcst, valid_loss

    def run(self):
        """
        Executes grid search algorithm with defined parameters
        :return:
        """

        random.seed(self.search_config.seed)
        print("Starting GRID search ...")

        for i in range(self.num_gens):
            print("Gen %i" % (i + 1))
            self._cur_pop = self._get_gen(i)
            assert isinstance(self._cur_pop, (list, tuple))
            self._train_population(i + 1)
            self.opt_hist.to_csv(os.path.join(self.opt_run_dir, 'opt_hist.csv'), sep=',', index=False)
