from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow.keras.optimizers as opt
import tensorflow.keras.optimizers.schedules as schedules


class Optimizers(object):

    def __init__(self, config):
        self.config = config
        self.optimizer = self.config.optimizer
        self.lr_decay = self.config.lr_decay
        self.learning_rate = self.get_learning_rate()

    def get_optimizer(self):
        """
        Returns tf.keras.optimizer based on config
        :return: optimizer
        """
        if self.optimizer == 'Adam':
            return opt.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == 'Adadelta':
            return opt.Adadelta(learning_rate=self.learning_rate)
        elif self.optimizer == 'RMSprop':
            return opt.RMSprop(learning_rate=self.learning_rate)
        elif self.optimizer == 'SGD':
            return opt.SGD(learning_rate=self.learning_rate, momentum=self.config.sgd_momentum)
        else:
            raise ValueError("%s optimizer not found in tf.keras.optimizers" % self.optimizer)

    def get_learning_rate(self):
        """
        Returns keras schedule for learning rate based on lr_schedule specified in the config
        :return: keras.optimizers.schedules
        """

        if self.config.lr_schedule == 'ExponentialDecay':
            lr = schedules.ExponentialDecay(self.config.learning_rate,
                                            decay_steps=self.config.decay_steps,
                                            decay_rate=self.lr_decay,
                                            staircase=True)
        elif self.config.lr_schedule == 'PolynomialDecay':
            lr = schedules.PolynomialDecay(self.config.learning_rate,
                                           self.config.decay_steps,
                                           self.config.end_learning_rate,
                                           power=self.config.decay_power)
        elif self.config.lr_schedule == 'PiecewiseConstantDecay':
            lr = schedules.PiecewiseConstantDecay(self.config.piecewise_lr_boundaries,
                                                  self.config.piecewise_lr_values)
        else:
            print("Invalid learning rate scheduler specified")
            raise ValueError

        return lr

