from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomUniform

from models.model_base_class import BaseModelClass
from model_utils.optimizers import Optimizers
from model_utils.initializers import Initializer
from base_config import get_configs
from data_processing import Dataset


class MLPLinearPointEstimate(BaseModelClass):
    """
    MLP Point Estimate compiled model with the specified architecture

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
        Builds a model based on the architecture defined in the configs

        The input received is already padded from the data processing module for variable sequence length.
        Making is used to keep track of padded elements in the tensor. Keras layers such as Cropping1D and Concatenate
        do not use masking, hence custom layer RemoveMask is used to strip masking information from the outputs for
        such layers.

        Architecture Logic for Multi Step Forecast -> Append the output of previous forecast step to the next one

        1. Concatenate last time step aux features with outputs as outputs only contain financial fields
        2. Concatenate the above output to the inputs and strip the first element in the sequence to keep the input
            shape consistent
        3. Repeat 1,2 for subsequent outputs


        :return: compiled keras model which outputs ((output_1, mask_1), (output_2, mask_2), ...) where _1 refers to
        the forecast step. For example _1 : 12 month forecast, _2 : 24 month forecast and so on
        """

        outputs = []

        # Masking information is only used by certain layers such as LSTM. Hence two copies of inputs are used, one for
        # propagating the mask and second for storing inputs which are used in operations such as Cropping1D and
        # concatenate.
        inputs = x = keras.Input(shape=(self.seq_len * self.n_inputs), name='input_financials')

        initializer = self.initializer.get_initializer()

        output_count = 1
        cur_output = layers.Dense(self.n_outputs,
                                  kernel_initializer=initializer,
                                  name='OUTPUT_%i' % output_count)(x)

        outputs.append(cur_output)

        model = keras.Model(inputs=inputs, outputs=outputs)

        return model


if __name__ == '__main__':

    config = get_configs()
    config.train = True
    config.scale_field = ''
    config.max_unrollings = 8
    config.forecast_n = 12
    config.forecast_steps = 2
    config.data_dir = '../../../datasets'
    config.datafile = 'sample_data.dat'

    D = Dataset(config)

    rnn_model = MLPLinearPointEstimate(config, D)
    m = rnn_model.model
    print(m.summary())

    plot_path = '../../../dev_test/model.png'
    keras.utils.plot_model(m, plot_path, show_shapes=True)

    # t_set = D.train_set
    # t_set = t_set.batch(batch_size=config.batch_size)
    # v_set = D.valid_set
    # v_set = v_set.batch(batch_size=config.batch_size)

    # for (batch_n, train_set_items) in enumerate(t_set):
    #     inp_idxs = train_set_items[0]
    #     tar_idxs = train_set_items[1]
    #     pre_metadata = train_set_items[2]
    #     inp, targets, metadata = D.get_batch(inp_idxs, tar_idxs, pre_metadata)

    #     with tf.GradientTape() as tape:
    #         preds = m(inp)

    #         # re-assign targets, preds as lists as required by loss function
    #         if config.forecast_steps == 1:
    #             preds = [preds]
    #             targets = [targets]

    #         assert isinstance(targets, (list, tuple))
    #         assert isinstance(preds, (list, tuple))
    #     break

