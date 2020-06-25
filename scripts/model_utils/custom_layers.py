from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers


class SoftPlus(layers.Layer):

    def __init__(self):
        super(SoftPlus, self).__init__()

    def call(self, inputs):
        return tf.math.maximum(tf.math.softplus(inputs), 1e-6)

