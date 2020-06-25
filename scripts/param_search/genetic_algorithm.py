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

from itertools import cycle
from param_search.opt_base_class import OptimizerBaseClass


class GeneticAlgorithm(OptimizerBaseClass):
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

        if self.search_config.doe_file:
            self.doe = pd.read_csv(self.search_config.doe_file, sep=',', dtype=object)
            self.NPE = self.doe.shape[0]
            self.pop_size = self.NPE

        # create experiment dir
        self.exp_dirname = os.path.join(self.search_config.experiments_dir, self.search_config.name)
        if not os.path.isdir(self.exp_dirname):
            os.mkdir(self.exp_dirname)

        # create opt_run dir
        self.opt_run_dir = os.path.join(self.exp_dirname, 'opt_run')
        if not os.path.isdir(self.opt_run_dir):
            os.mkdir(self.opt_run_dir)

        # Initialize random population
        self._cur_pop = self._init_population()

    def _init_population(self):
        """
        Creates initial random population of pop_size
        :return: list of configs
        """
        random.seed(self.search_config.seed)
        pop = []

        for i in range(self.pop_size):
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

            if self.search_config.doe_file:
                member = self.doe.iloc[i]
                # update variables from the member pandas series
                for var in member.index:
                    config.__dict__["__configs"][var] = member[var]

            else:
                # randomly assign tunable variables a value
                for var in self.tunable_vars:
                    if var not in self.vars_keep_type:  # vars that need to be list type
                        config.__dict__["__configs"][var] = random.choice(self.search_config_dict[var])

            for k, v in config.__dict__["__configs"].items():
                if k not in self.vars_keep_type:
                    assert v is None or isinstance(v, (int, np.int64, np.int32,
                                                       float, np.float64, np.float32,
                                                       str,
                                                       bool, np.bool, np.bool_)), \
                        "All configuration parameters should have singular value. Received %g for param %s" % (v, k)

            pop.append(config)

        return pop

    @staticmethod
    def _reset_member_ids(pop):
        """
        Resets the id numbers for the population to begin with 1. To be used when  initial population is loaded
        externally
        :param pop: population whose member ids need to be reset. type: list
        :return:
        """
        mem_id = 1

        for p in pop:
            p.__dict__["__configs"]['member_id'] = mem_id
            mem_id += 1
        return pop

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

        return results

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
                    if isinstance(v, (int, np.int64, np.int32,
                                      float, np.float64, np.float32,
                                      str,
                                      bool, np.bool, np.bool_)):
                        f.write('--{}       {} \n'.format(k, v))
                    elif isinstance(v, (list, tuple)):
                        f.write('--{}       '.format(k) + '-'.join('{}'.format(j) for j in v) + '\n')

        os.system("CUDA_VISIBLE_DEVICES=%s python deep_quant.py --config=%s" % (str(config.default_gpu)[-1],
                                                                                config_path))

        valid_loss, valid_loss_fcst = self._read_results(config)

        # update member objective result
        self.opt_hist.loc[config.member_id, 'objective'] = valid_loss_fcst
        self.opt_hist.loc[config.member_id, 'valid_mse_fcst'] = valid_loss_fcst
        self.opt_hist.loc[config.member_id, 'valid_mse'] = valid_loss

        return valid_loss, valid_loss_fcst, config

    @staticmethod
    def _read_results(config):
        """
        Returns the last epoch's result from training log
        :param config: case config; type: config object
        :return: validation loss
        """

        # TODO: Add uq loss as in doe.py

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

        return valid_loss, valid_loss_fcst

    def _get_next_gen(self, results):
        """
        returns next generation's population after crossover and mutation
        :param results: results of current generation. type: list [(objective 1, config 1), (objective 2, config 2)]
        :return: list of configs
        """

        assert isinstance(results, (list, tuple))

        results = sorted(results, key=lambda x: x[0])  # sort by objective function

        parents = [x[-1] for x in results[:len(results) // 2]]

        self._cur_pop = []

        for i in range(self.pop_size):
            # cross over
            random.shuffle(parents)
            p1, p2 = parents[0], parents[1]
            child = self._cross_over(p1, p2)
            # mutate
            if random.random() <= self.mutate_rate:
                child = self._mutate(child)

            self._cur_pop.append(child)

        return self._cur_pop

    def _cross_over(self, p1, p2):
        """
        Cross over action in genetic algorithm using two parents to produce a child
        :param p1: parent 1; type: config object
        :param p2: parent 2; type: config object
        :return: child; type: config object
        """
        self._id += 1
        child = copy.deepcopy(p1)
        flags = list(child.__dict__["__configs"].keys())

        for flag in flags:
            if random.random() >= 0.5:
                child.__dict__["__configs"][flag] = p2.__dict__["__configs"][flag]

        child.__dict__["__configs"]['member_id'] = self._id
        child.__dict__["__configs"]["model_dir"] = os.path.join(self.search_config.name,
                                                                'opt_run',
                                                                self.search_config.name + \
                                                                '-' + str(self._id))

        # assign gpu
        child.__dict__['__configs']['default_gpu'] = '/gpu:' + str(next(self.gpu_id_iterable))

        return child

    def _mutate(self, mem):
        """
        Mutation action in genetic algorithm
        :param mem: member of a population; type: config object
        :return: mem
        """
        mutate_flag = random.choice(self.tunable_vars)
        mem.__dict__["__configs"][mutate_flag] = random.choice(self.search_config_dict[mutate_flag])
        return mem

    def run(self):
        """
        Executes genetic algorithm with defined parameters
        :return:
        """

        random.seed(self.search_config.seed)
        print("Starting GA search ...")

        # Override random initial pop if initial pop is provided
        if self.init_pop:
            self._cur_pop = pickle.load(open(self.init_pop, 'rb'))

        assert isinstance(self._cur_pop, (list, tuple))

        for i in range(self.num_gens):
            gen = i + 1
            print("Gen %i" % gen)

            if self.search_config.save_latest_pop:
                latest_pop_dirname = os.path.join(self.exp_dirname, '_latest_pop')
                if not os.path.isdir(latest_pop_dirname):
                    os.mkdir(latest_pop_dirname)

                pickle.dump(self._cur_pop, open(os.path.join(latest_pop_dirname, 'latest_pop.pkl'), 'wb'))

            pop_results = self._train_population(gen)

            # TODO: Confirm if diversity calculation is necessary

            self._cur_pop = self._get_next_gen(pop_results)

            self.opt_hist.to_csv(os.path.join(self.opt_run_dir, 'opt_hist.csv'), sep=',', index=False)

