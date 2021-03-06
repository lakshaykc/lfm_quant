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
from model_utils.custom_layers import SoftPlus
from base_config import get_configs
from data_processing import Dataset


class MLPUqRangeEstimate(BaseModelClass):
    """
    MLP UQ Range Estimate compiled model with the specified architecture

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
        Builds a mlp uq estimate model based on the architecture defined in the configs

        The input received is already padded from the data processing module for variable sequence length.
        Making is used to keep track of padded elements in the tensor. Keras layers such as Cropping1D and Concatenate
        do not use masking, hence custom layer RemoveMask is used to strip masking information from the outputs for
        such layers.

        Architecture Logic for Multi Step Forecast -> Append the output of previous forecast step to the next one

        1. Concatenate last time step aux features with outputs as outputs only contain financial fields
        2. Concatenate the above output to the inputs and strip the first element in the sequence to keep the input
            shape consistent
        3. Repeat 1,2 for subsequent outputs


        :return: compiled keras model which outputs (output_1, output_2, ...) where _1 refers to
        the forecast step. For example _1 : 12 month forecast, _2 : 24 month forecast and so on
        """

        outputs = []

        # Masking information is only used by certain layers such as LSTM. Hence two copies of inputs are used, one for
        # propagating the mask and second for storing inputs which are used in operations such as Cropping1D and
        # concatenate.
        inputs = x = keras.Input(shape=(self.seq_len * self.n_inputs), name='input_financials')
        prev_input = inputs

        # last_time_step_aux = self.get_last_time_step_aux(x)

        initializer = self.initializer.get_initializer()

        hidden_layer_count = 0
        output_count = 0

        for i in range(self.n_layers):
            hidden_layer_count += 1
            x = layers.Dense(self.n_hidden_units,
                             kernel_initializer=initializer,
                             kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_alpha),
                             kernel_constraint=MaxNorm(self.config.max_norm),
                             use_bias=False,
                             name='dense_%i' % hidden_layer_count)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(rate=self.config.dropout)(x, training=True)

        output_count += 1
        # outputs for target values
        cur_output_tar = layers.Dense(self.n_outputs, name='OUTPUT_TARGET_%i' % output_count)(x)
        # outputs for variances of the target values
        cur_output_var = layers.Dense(self.n_outputs, name='OUTPUT_VARIANCE_%i' % output_count)(x)
        cur_output_var = SoftPlus()(cur_output_var)

        outputs.append(cur_output_tar)
        outputs.append(cur_output_var)

        for fcst_step in range(1, self.forecast_steps):
            print("Multi-step forecast not implemented for MLP")
            raise NotImplementedError
            # # output_count, lstm_count keep track of layer ids. output_count and fcst_step are not the same as one
            # # fcst_step could have multiple outputs.
            # output_count += 1
            # cur_output = outputs[-2]  # last target output
            # last_time_step_fin = self.get_last_time_step(cur_output, output_count)
            # # Combine latest prediction with last available aux features to make the input shape compatible
            # last_time_step = layers.concatenate([last_time_step_fin, last_time_step_aux], axis=2,
            #                                     name='concat_fin_aux_%i' % fcst_step)
            # # combine latest prediction with input sequence
            # cur_input = layers.concatenate([prev_input, last_time_step], axis=1,
            #                                name='combine_input_w_last_pred_%i' % fcst_step)
            # cur_input = layers.Cropping1D(cropping=(1, 0), name='updated_input_w_last_pred_%i' % fcst_step)(cur_input)
            # prev_input = cur_input
            # # Add LSTM layer for intermediary prediction
            # hidden_layer_count += 1
            # dense_intm = layers.Dense(self.n_hidden_units,
            #                           activation='relu',
            #                           name='dense_%i' % hidden_layer_count)(cur_input)
            # dense_intm = layers.Dropout(rate=self.config.dropout)(dense_intm, training=self.config.train)
            # # outputs for target values
            # outputs.append(layers.Dense(self.n_outputs, name='OUTPUT_TARGET_%i' % output_count)(dense_intm))
            # # outputs for variances for the target values
            # outputs.append(layers.Dense(self.n_outputs, name='OUTPUT_VARIANCE_%i' % output_count)(dense_intm))

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

    rnn_model = MLPUqRangeEstimate(config, D)
    m = rnn_model.model
    print(m.summary())

    plot_path = '../../../dev_test/model.png'
    keras.utils.plot_model(m, plot_path, show_shapes=True)

    # t_set = D.train_set
    # t_set = t_set.batch(batch_size=config.batch_size)
    # v_set = D.valid_set
    # v_set = v_set.batch(batch_size=config.batch_size)
    #
    # for (batch_n, train_set_items) in enumerate(t_set):
    #     inp_idxs = train_set_items[0]
    #     tar_idxs = train_set_items[1]
    #     pre_metadata = train_set_items[2]
    #     inp, targets, metadata = D.get_batch(inp_idxs, tar_idxs, pre_metadata)
    #
    #     with tf.GradientTape() as tape:
    #         preds = m(inp)
    #
    #         # re-assign targets, preds as lists as required by loss function
    #         if config.forecast_steps == 1:
    #             preds = [preds]
    #             targets = [targets]
    #
    #         assert isinstance(targets, (list, tuple))
    #         assert isinstance(preds, (list, tuple))
    #     break
