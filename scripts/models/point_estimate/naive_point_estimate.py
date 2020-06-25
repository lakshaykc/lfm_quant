from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.constraints import MaxNorm

from models.model_base_class import BaseModelClass
from model_utils.optimizers import Optimizers
from model_utils.initializers import Initializer
from base_config import get_configs
from data_processing import Dataset


class NaivePointEstimate(BaseModelClass):
    """
    Naive Point Estimate compiled model with the specified architecture

    Builds a keras model
    """

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.seq_len = self.dataset.seq_len
        self.n_inputs = self.dataset.n_inputs
        self.n_outputs = self.dataset.n_outputs
        self.forecast_steps = self.config.forecast_steps
        self.n_layers = self.config.num_layers
        self.n_hidden_units = self.config.num_hidden
        self.opt = Optimizers(self.config)
        self.initializer = Initializer(self.config)
        super().__init__(self.seq_len, self.n_inputs, self.n_outputs)

        # Build model
        self.model = self._build_model()

    def _build_model(self):
        """
        """

        inputs = keras.Input(shape=(self.seq_len * self.n_inputs), name='input_financials')

        y = layers.Reshape(target_shape=(self.seq_len, self.n_inputs))(inputs)

        # crop all except the last time step
        last_step = layers.Cropping1D(cropping=(self.seq_len - 1, 0), name='last_time_step')(y)
        # crop all except the financial features which are last self.n_outputs features
        # Cropping1D only allows for cropping on axis=0 and hence the use of reshaping
        last_step = layers.Reshape(target_shape=(self.n_inputs, 1))(last_step)
        last_step = layers.Cropping1D(cropping=(0, self.n_inputs - self.n_outputs), name='financial_features')(last_step)
        last_step = layers.Reshape(target_shape=(1, self.n_outputs))(last_step)
        last_step = layers.Reshape(target_shape=(self.n_outputs,))(last_step)

        model = keras.Model(inputs=inputs, outputs=last_step)

        return model


if __name__ == '__main__':

    config = get_configs()
    config.train = False
    config.scale_field = ''
    config.max_unrollings = 8
    config.forecast_n = 12
    config.forecast_steps = 1
    config.data_dir = '../../../datasets'
    config.datafile = 'sample_data_testing_2.dat'
    config.nn_type = 'NaivePointEstimate'
    config.batch_size = 1

    D = Dataset(config)

    rnn_model = NaivePointEstimate(config, D)
    m = rnn_model.model
    print(m.summary())

    #plot_path = '../../../dev_test/model.png'
    #keras.utils.plot_model(m, plot_path, show_shapes=True)

    t_set = D.test_set
    t_set = t_set.batch(batch_size=config.batch_size)
    # v_set = D.valid_set
    # v_set = v_set.batch(batch_size=config.batch_size)

    for (batch_n, train_set_items) in enumerate(t_set):
        inp_idxs = train_set_items[0]
        tar_idxs = train_set_items[1]
        pre_metadata = train_set_items[2]
        inp, targets, metadata = D.get_batch(inp_idxs, tar_idxs, pre_metadata)

        preds = m.predict(inp)

        print(inp)
        print(targets)
        print(preds)
        break

    #     with tf.GradientTape() as tape:
    #         preds = m(inp)

    #         # re-assign targets, preds as lists as required by loss function
    #         if config.forecast_steps == 1:
    #             preds = [preds]
    #             targets = [targets]

    #         assert isinstance(targets, (list, tuple))
    #         assert isinstance(preds, (list, tuple))
    #     break

