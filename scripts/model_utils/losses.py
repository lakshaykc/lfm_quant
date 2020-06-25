from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.keras.losses import Huber

from base_config import get_configs


class Losses(object):

    def __init__(self, config, target_idx):
        self.config = config
        self.target_idx = target_idx

    def weight_adjusted_mse(self, y_true, y_pred, is_validation=False):
        """
        Returns linearly weighted mse values using forecast steps weights
        :param y_true: [true_labels] ; list
        :param y_pred: [predictions] ; list
        :return: loss
        """
        losses_point_est = []
        mse_point_est = []

        assert self.config.forecast_steps > 0, 'forecasts_steps should be a positive integer. %i was provided' % \
                                               self.config.forecast_steps
        assert isinstance(y_true, (list, tuple)), \
            'arguments to loss function need to be a list [y_true], [y_true1, y_true2, ..]'
        assert isinstance(y_pred, (list, tuple)), \
            'arguments to loss function need to be a list [y_pred], [y_pred1, y_pred2, ..]'

        for i in range(self.config.forecast_steps):
            loss, mse = self._get_loss_point_estimate(y_true[i], y_pred[i], is_validation)
            losses_point_est.append(loss)
            mse_point_est.append(mse)

        assert len(losses_point_est) == len(self.config.forecast_steps_weights)
        assert len(mse_point_est) == len(self.config.forecast_steps_weights)

        if len(losses_point_est) == 1:
            self.config.forecast_steps_weights = [1.0]

        weighted_loss = sum([loss * weight for loss, weight in zip(losses_point_est,
                                                                   self.config.forecast_steps_weights)])

        weighted_mse = sum([mse * weight for mse, weight in zip(mse_point_est,
                                                                self.config.forecast_steps_weights)])

        return weighted_loss, weighted_mse

    def _get_loss_point_estimate(self, y_true, y_pred, is_validation=False):
        """
        Returns mean squared error adjusted for weights for target field (such as oiadpq) and last time
        steps in the sequence.

        The last time step prediction and target field prediction have different sensitivities

        $$Loss = target_lambda*(mse_last_time_step[target_field] +
                (1 - target_lambda)(rnn_lambda*mse_last_time_step + (1 - rnn_lambda)*mse_all_time_steps)) $$

        :param y_true: true_labels
        :param y_pred: predictions
        :return: loss
        """
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        mask = ~tf.reduce_all(tf.equal(y_true, tf.constant(0.0, dtype=y_true.dtype)), axis=-1)
        mask = tf.cast(mask, dtype=y_true.dtype)

        y_pred = tf.multiply(y_pred, tf.expand_dims(mask, -1))

        # RNN MSE
        if 'RNN' in self.config.nn_type:
            # tf.expand_dims is used to keep the shape consistent as num_outputs is used within masking operation
            last_time_step_pred = tf.expand_dims(y_pred[:, -1, :], 1)
            last_time_step_true = tf.expand_dims(y_true[:, -1, :], 1)

            last_time_step_pred_tar = tf.expand_dims(last_time_step_pred[:, -1, self.target_idx], -1)
            last_time_step_true_tar = tf.expand_dims(last_time_step_true[:, -1, self.target_idx], -1)

            # last time step target loss
            mse_0 = tf.reduce_mean(math_ops.squared_difference(last_time_step_true_tar, last_time_step_pred_tar))

            # last time step loss
            mse_1 = tf.reduce_mean(math_ops.squared_difference(last_time_step_true, last_time_step_pred))

            # all time steps loss
            mse_2 = self._get_mse(y_true, y_pred, mask)

            p1 = self.config.target_lambda
            p2 = self.config.rnn_lambda

            return p1 * mse_0 + (1.0 - p1) * (p2 * mse_1 + (1 - p2) * mse_2), mse_0

        elif 'MLP' in self.config.nn_type:
            # target field
            pred_tar = y_pred[:, self.target_idx]
            true_tar = y_true[:, self.target_idx]

            if self.config.huber_loss and not is_validation:
                h = Huber(delta=self.config.huber_delta)
                mse_0 = h(true_tar, pred_tar)
                mse_all = h(y_true, y_pred)

            else:
                mse_0 = tf.reduce_mean(math_ops.squared_difference(true_tar, pred_tar))
                # mse wrt all outputs
                mse_all = tf.reduce_mean(math_ops.squared_difference(y_true, y_pred))

            return self.config.target_lambda * mse_0 + (1. - self.config.target_lambda) * mse_all, mse_0

        else:
            print("%s NN type not implemented" % self.config.nn_type)
            raise NotImplementedError

    @staticmethod
    def _get_mse(y_true, y_pred, mask):
        """
        Calculates mean squared error using mask. Mask is of shape (batch_size, seq_len) i.e if a time step in a
        batch is masked or not. Hence mask is multiplied by number of outputs.
        :param y_true:
        :param y_pred:
        :param mask:
        :return:
        """
        diff = tf.reduce_sum(math_ops.squared_difference(y_pred, y_true))
        mask_sum = tf.reduce_sum(tf.cast(mask, tf.float32)) * y_true.shape[-1]
        mask_sum = tf.cast(mask_sum, dtype=diff.dtype)

        return diff / mask_sum

    def weight_adjusted_uq_loss(self, y_true, y_pred, y_var):
        """
        returns linearly weighted UQ loss using forecast step weights
        :param y_true: [true labels] ; type:list
        :param y_pred: [target predictions] ; type:list
        :param y_var: [variance predictions] ; type:list
        :return: uq_loss, mse
        """
        assert self.config.UQ, "weight_adjusted_mse is only available for uq range estimate models. UQ should be True"
        uq_losses = []
        uq_losses_last_tar = []
        mses = []

        assert self.config.forecast_steps > 0, 'forecasts_steps should be a positive integer. %i was provided' % \
                                               self.config.forecast_steps
        assert isinstance(y_true, (list, tuple)), \
            'arguments to loss function need to be a list [y_true], [y_true1, y_true2, ..]'
        assert isinstance(y_pred, (list, tuple)), \
            'arguments to loss function need to be a list [y_pred], [y_pred1, y_pred2, ..]'
        assert isinstance(y_var, (list, tuple)), \
            'arguments to loss function need to be a list [y_var], [y_var1, y_var2, ..]'

        for i in range(self.config.forecast_steps):
            uq_loss, uq_loss_last_tar, mse = self._get_loss_uq_estimate(y_true[i], y_pred[i], y_var[i])
            uq_losses.append(uq_loss)
            uq_losses_last_tar.append(uq_loss_last_tar)
            mses.append(mse)

        assert len(uq_losses) == len(self.config.forecast_steps_weights)
        assert len(uq_losses_last_tar) == len(self.config.forecast_steps_weights)
        assert len(mses) == len(self.config.forecast_steps_weights)

        if len(uq_losses) == 1:
            self.config.forecast_steps_weights = [1.0]

        w_uq_loss = sum([loss * weight for loss, weight in zip(uq_losses, self.config.forecast_steps_weights)])
        w_uq_loss_last_tar = sum([loss * weight for loss, weight in zip(uq_losses_last_tar,
                                                                        self.config.forecast_steps_weights)])
        w_mse = sum([mse * weight for mse, weight in zip(mses, self.config.forecast_steps_weights)])

        return w_uq_loss, w_uq_loss_last_tar, w_mse

    def _get_loss_uq_estimate(self, y_true, y_pred, y_var):
        """
        Returns UQ loss adjusted for weights for target field (such as oiadpq) and last time
        steps in the sequence.

        The last time step prediction and target field prediction have different sensitivities

        $$Loss = target_lambda*(uq_loss_last_time_step[target_field] +
                (1 - target_lambda)(rnn_lambda*uq_loss_last_time_step + (1 - rnn_lambda)*uq_loss_all_time_steps)) $$

        :param y_true: true_labels
        :param y_pred: predictions
        :param y_var: variances
        :return: loss
        """

        y_pred = ops.convert_to_tensor(y_pred)
        y_var = ops.convert_to_tensor(y_var)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        mask = ~tf.reduce_all(tf.equal(y_true, tf.constant(0.0, dtype=y_true.dtype)), axis=-1)
        mask = tf.cast(mask, dtype=y_true.dtype)

        y_pred = tf.multiply(y_pred, tf.expand_dims(mask, -1))
        y_var = tf.multiply(y_var, tf.expand_dims(mask, -1))

        if 'RNN' in self.config.nn_type:
            # tf.expand_dims is used to keep the shape consistent as num_outputs is used within masking operation
            last_time_step_pred = tf.expand_dims(y_pred[:, -1, :], 1)
            last_time_step_var = tf.expand_dims(y_var[:, -1, :], 1)
            last_time_step_true = tf.expand_dims(y_true[:, -1, :], 1)

            last_time_step_pred_tar = tf.expand_dims(last_time_step_pred[:, -1, self.target_idx], -1)
            last_time_step_var_tar = tf.expand_dims(last_time_step_var[:, -1, self.target_idx], -1)
            last_time_step_true_tar = tf.expand_dims(last_time_step_true[:, -1, self.target_idx], -1)

            # last time step target loss
            uq_loss_0 = self._get_uq_loss(last_time_step_true_tar,
                                          last_time_step_pred_tar,
                                          last_time_step_var_tar,
                                          mask[:, -1])

            mse_0 = tf.reduce_mean(math_ops.squared_difference(last_time_step_true_tar, last_time_step_pred_tar))

            # last time step loss
            uq_loss_1 = self._get_uq_loss(last_time_step_true,
                                          last_time_step_pred,
                                          last_time_step_var,
                                          mask[:, -1])

            # all time steps loss
            uq_loss_2 = self._get_uq_loss(y_true, y_pred, y_var, mask)

            p1 = self.config.target_lambda
            p2 = self.config.rnn_lambda
            return p1 * uq_loss_0 + (1.0 - p1) * (p2 * uq_loss_1 + (1 - p2) * uq_loss_2), uq_loss_0, mse_0

        elif 'MLP' in self.config.nn_type:
            # target field
            pred_tar = y_pred[:, self.target_idx]
            var_tar = y_var[:, self.target_idx]
            true_tar = y_true[:, self.target_idx]

            uq_loss_0 = self._get_uq_loss(true_tar, pred_tar, var_tar, None)

            mse_0 = tf.reduce_mean(math_ops.squared_difference(true_tar, pred_tar))

            # uq_loss wrt all outputs
            uq_loss_all = self._get_uq_loss(y_true, y_pred, y_var, None)

            return self.config.target_lambda * uq_loss_0 + (1. - self.config.target_lambda) * uq_loss_all, \
                   uq_loss_0, mse_0

        else:
            print("%s NN type not implemented" % self.config.nn_type)
            raise NotImplementedError

    def _get_uq_loss(self, y_true, y_pred, y_var, mask):
        """
        Calculates the uq loss function using mask. Mask is of shape (batch_size, seq_len) i.e if a time step in a
        batch is masked or not. Hence mask is multiplied by number of outputs.
        :param y_true:
        :param y_pred:
        :param y_var:
        :param mask:
        :return:
        """
        diff = math_ops.squared_difference(y_pred, y_true)

        if 'RNN' in self.config.nn_type:
            mask_sum = tf.reduce_sum(tf.cast(mask, tf.float32)) * y_true.shape[-1]
            mask_sum = tf.cast(mask_sum, dtype=diff.dtype)

            loss = tf.multiply(diff, tf.divide(1., y_var)) + tf.math.log(y_var)
            uq_loss = tf.reduce_sum(loss) / mask_sum

        elif 'MLP' in self.config.nn_type:
            assert mask is None
            uq_loss = tf.multiply(diff, tf.divide(1., y_var)) + tf.math.log(y_var)
            uq_loss = tf.reduce_mean(uq_loss)

        else:
            print("loss function for %s not implemented" % self.config.nn_type)
            raise NotImplementedError

        return uq_loss


if __name__ == '__main__':
    config = get_configs()
    config.forecast_steps = 1
    config.forecast_steps_weights = [1.0]
    config.target_lambda = 1.0
    config.rnn_lambda = 0.0

    losses = Losses(config, 2)

    y_true = tf.constant([[[0, 0, 0], [0, 0, 0], [4, 5, 6], [7, 8, 9], [1, 2, 3]],
                          [[0, 0, 0], [0, 0, 0], [1, 2, 3], [4, 5, 6], [7, 8, 9]]],
                         dtype=tf.float32)

    y_pred = np.ones(shape=y_true.shape)
    # y_pred[:, 0, :] = 0.0
    # y_pred[:, 1, :] = 0.0
    y_pred = tf.constant(y_pred, dtype=tf.float32)

    print("y_true:")
    print(y_true)
    print('y_pred')
    print(y_pred)

    print(losses.weight_adjusted_mse([y_true], [y_pred]))
