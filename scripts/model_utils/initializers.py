from __future__ import absolute_import, division, print_function, unicode_literals


from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.initializers import GlorotUniform


class Initializer(object):

    def __init__(self, config):
        self.config = config

    def get_initializer(self):

        if self.config.use_custom_init:
            return RandomUniform(-self.config.init_scale, self.config.init_scale, seed=self.config.seed)
        else:
            if self.config.initializer == 'GlorotNormal':
                return GlorotNormal(seed=self.config.seed)
            elif self.config.initializer == 'GlorotUniform':
                return GlorotUniform(seed=self.config.seed)
            else:
                raise NotImplementedError
