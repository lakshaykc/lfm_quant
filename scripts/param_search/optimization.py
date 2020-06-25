from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .genetic_algorithm import GeneticAlgorithm
from.grid_search import GridSearch
from.doe import DOE


class HPOptimization(object):
    """
    Executes hyper param optimization
    """
    def __init__(self, search_config):
        self.search_config = search_config

    def __call__(self):
        if self.search_config.search_algorithm == 'genetic':
            ga = GeneticAlgorithm(self.search_config)
            ga.run()
        elif self.search_config.search_algorithm == 'grid_search':
            gs = GridSearch(self.search_config)
            gs.run()
        elif self.search_config.search_algorithm == 'doe':
            doe = DOE(self.search_config)
            doe.run()
        else:
            print(self.search_config.search_algorithm)
            raise NotImplementedError
