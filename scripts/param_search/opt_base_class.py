from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class OptimizerBaseClass(object):
    """
    Base class for hyper parameter optimization algorithms
    """

    def __init__(self, search_config):
        self.search_config = search_config
        self.objective = self.search_config.objective

        # Variables
        self.search_config_dict = self.search_config.__dict__["__configs"]
        self.all_vars, self.constant_vars, self.tunable_vars = [], [], []

        for k, v in self.search_config_dict.items():
            self.all_vars.append(k)
            if isinstance(v, (list, tuple)) and len(v) > 1:
                self.tunable_vars.append(k)
            elif isinstance(v, (list, tuple)) and len(v) <= 1:
                self.constant_vars.append(k)

        self.opt_history = None

        self.NPE = self.search_config.NPE
        self.num_gpu = self.search_config.num_gpu
        self.num_gens = self.search_config.generations
        self.pop_size = self.search_config.pop_size
        self.init_pop = self.search_config.init_pop
        self.mutate_rate = self.search_config.mutate_rate

        self.vars_keep_type = ['forecast_steps_weights']
