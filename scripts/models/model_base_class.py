from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
from tensorflow.keras import layers


class BaseModelClass(object):
    """
    Base class for models to provide data from the last time step of the sequence.
    """

    def __init__(self, seq_len, n_inputs, n_outputs):
        self.seq_len = seq_len
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def get_last_time_step_aux(self, sequence):
        """
        Returns aux features of last time step of `sequence`. ObjectType: `keras.layers`

        The model outputs fundamental features only. Hence, predicted fundamental features along with last available
        aux features are used to predict fundamentals further in the future than the last prediction.

        keras.layers.Cropping1D operation is used.

        :param sequence: objectType: keras.layer
        :return: keras.layer of last time step aux features
        """
        # crop all except the last time step
        last_step = layers.Cropping1D(cropping=(self.seq_len - 1, 0), name='last_time_step_aux')(sequence)

        # crop all except the aux features which are last self.n_outputs features
        # Cropping1D only allows for cropping on axis=0 and hence the use of reshaping
        last_step = layers.Reshape(target_shape=(self.n_inputs, 1))(last_step)
        last_step = layers.Cropping1D(cropping=(self.n_outputs, 0), name='aux_features')(last_step)
        last_step = layers.Reshape(target_shape=(1, self.n_inputs - self.n_outputs))(last_step)

        return last_step

    def get_last_time_step(self, sequence, count):
        """
        Returns all features of the last time step of `sequence`. ObjectType: `keras.layers`

        keras.layers.Cropping1d operation is used
        :param sequence: keras.layer sequence
        :param count: output count to keep tracking of layer name
        :return: last time step of the sequence (layer)
        """

        return layers.Cropping1D(cropping=(self.seq_len - 1, 0), name='last_time_step_%i' % count)(sequence)
