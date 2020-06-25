from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import platform
import time
import numpy as np
import pandas as pd
import copy
# import matplotlb.pyplot as plt
import tensorflow as tf
from collections import defaultdict
from tensorflow.python.framework import ops
import pathlib
import random
from model_utils.optimizers import Optimizers
from model_utils.model import Model
from model_utils.losses import Losses
from base_config import get_configs
from data_processing import Dataset

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)


class Train(object):

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.dataset.generate_dataset()
        self.model = Model(self.config, self.dataset).get_model()
        self.target_index = self.dataset.target_index
        self.optimizer = Optimizers(self.config).get_optimizer()
        self.losses = Losses(self.config, self.target_index)

        # Load data
        self.train_set = self.dataset.train_set
        self.valid_set = self.dataset.valid_set

        self.train_set = self.train_set.shuffle(buffer_size=10000, seed=self.config.seed)
        self.train_set = self.train_set.batch(batch_size=self.config.batch_size)
        self.valid_set = self.valid_set.batch(batch_size=self.config.batch_size)

        # Create model level directories for chkpts and logs
        self.model_dir = os.path.join(self.config.experiments_dir, self.config.model_dir)
        self.chkpts_dir = os.path.join(self.model_dir, 'chkpts')
        self.train_log_dir = os.path.join(self.model_dir, 'train_log')

        train_dirs = [self.model_dir, self.chkpts_dir, self.train_log_dir]
        print(train_dirs)
        for dirname in train_dirs:
            if not os.path.isdir(dirname):
                print("Creating dir %s" % dirname)
                pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)

        self.min_valid_mse = np.inf
        self.min_valid_uq_loss = np.inf

        # initialize grad norm
        self._grad_norm = tf.constant(1.0)

        # Create Batches
        print("Creating batches ...")
        self._batches = []
        for (batch_n, train_set_items) in enumerate(self.train_set):
            inp_idxs = train_set_items[0]
            tar_idxs = train_set_items[1]
            pre_metadata = train_set_items[2]
            inp, target, metadata = self.dataset.get_batch(inp_idxs, tar_idxs, pre_metadata)
            self._batches.append((inp, target, metadata))

        self._valid_batches = []
        for (batch_n, train_set_items) in enumerate(self.valid_set):
            inp_idxs = train_set_items[0]
            tar_idxs = train_set_items[1]
            pre_metadata = train_set_items[2]
            inp_val, target_val, metadata_val = self.dataset.get_batch(inp_idxs, tar_idxs, pre_metadata)
            self._valid_batches.append((inp_val, target_val, metadata_val))

    def train(self):

        # load from saved weights
        if self.config.load_saved_weights:
            checkpoint_path = os.path.join(self.chkpts_dir, "chkpt")
            self.model.load_weights(checkpoint_path)

        print("Training in progress ...")
        # Training step
        epochs = self.config.max_epoch

        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(self.chkpts_dir, "chkpt")

        # initialize logging
        train_logs_batch = defaultdict(list)
        train_logs_epoch = defaultdict(list)
        self.model.save_weights(checkpoint_prefix)

        start = time.time()
        for epoch in range(epochs):
            # initializing the hidden state at the start of every epoch
            # initially hidden is None
            hidden = self.model.reset_states()

            mse_steps, uq_loss_steps = [], []

            # for (batch_n, train_set_items) in enumerate(self.train_set):
            #     inp_idxs = train_set_items[0]
            #     tar_idxs = train_set_items[1]
            #     pre_metadata = train_set_items[2]
            #     inp, target, metadata = self.dataset.get_batch(inp_idxs, tar_idxs, pre_metadata)

            random.shuffle(self._batches)

            for (batch_n, cur_batch) in enumerate(self._batches):
                inp = cur_batch[0]
                target = cur_batch[1]
                metadata = cur_batch[2]

                if self.config.UQ:
                    uq_loss, mse = self._train_step_uq_range(inp, target)
                else:
                    mse = self._train_step_point(inp, target)
                    uq_loss = None

                # Collect mse, uq_loss for the current epoch
                mse_steps.append(mse)
                uq_loss_steps.append(uq_loss)

                # Log batch level metrics
                if batch_n % self.config.logging_interval == 0:
                    train_logs_batch['batch_n'].append(batch_n)
                    train_logs_batch['time'].append(time.time() - start)
                    train_logs_batch['mse'].append(sum(mse_steps) / len(mse_steps))
                    if uq_loss:
                        train_logs_batch['uq_loss'].append(sum(uq_loss_steps) / len(uq_loss_steps))
                    else:
                        train_logs_batch['uq_loss'].append(None)

                    # validation metrics are only computed at the end of a epoch
                    train_logs_batch['valid_mse'].append(None)
                    train_logs_batch['valid_uq_loss'].append(None)

                    # write files, plots
                    self._write_train_logs(train_logs_batch, 'train-logs-batch')

            if epoch % self.config.epoch_logging_interval == 0:
                # validation metrics
                if self.config.UQ:
                    valid_uq_loss, valid_mse, valid_mse_fcst = self._validation_metrics_uq_range_estimate()
                else:
                    valid_uq_loss, valid_mse, valid_mse_fcst = self._validation_metrics_point_estimate()
                # log epoch level metrics
                train_logs_epoch['epoch'].append(epoch)
                train_logs_epoch['time'].append(time.time() - start)
                train_logs_epoch['mse'].append(sum(mse_steps) / len(mse_steps))
                if uq_loss:
                    train_logs_epoch['uq_loss'].append(sum(uq_loss_steps) / len(uq_loss_steps))
                else:
                    train_logs_epoch['uq_loss'].append(None)
                train_logs_epoch['valid_mse'].append(valid_mse)
                train_logs_epoch['valid_uq_loss'].append(valid_uq_loss)
                train_logs_epoch['valid_mse_fcst'].append(valid_mse_fcst)

                self._write_train_logs(train_logs_epoch, 'train-logs-epoch')

                # saving criteria
                if self._save_criteria(train_logs_epoch):
                    self.model.save_weights(checkpoint_prefix)

                if self._stop_criteria(train_logs_epoch):
                    break

        return valid_mse

    def _train_step_point(self, inp, targets):
        assert not self.config.UQ

        with tf.GradientTape() as tape:
            preds = self.model(inp)
            # re-assign targets, preds as lists as required by loss function
            if self.config.forecast_steps == 1:
                preds = [preds]
                targets = [targets]

            assert isinstance(targets, (list, tuple))
            assert isinstance(preds, (list, tuple))

            loss, mse = self.losses.weight_adjusted_mse(targets, preds)
        grads = tape.gradient(loss, self.model.trainable_variables)

        # gradient clipping
        if self.config.max_grad_norm > 0:
            grads, self._grad_norm = tf.clip_by_global_norm(grads, self.config.max_grad_norm)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return mse.numpy()

    def _train_step_uq_range(self, inp, targets):
        assert self.config.UQ

        with tf.GradientTape() as tape:
            preds = self.model(inp)
            tar_preds = preds[0::2]  # target preds
            var_preds = preds[1::2]  # variance preds

            if self.config.forecast_steps == 1:
                targets = [targets]

            assert isinstance(targets, (list, tuple))
            assert isinstance(tar_preds, (list, tuple))
            assert isinstance(var_preds, (list, tuple))

            uq_loss, uq_loss_last_tar, mse = self.losses.weight_adjusted_uq_loss(targets, tar_preds, var_preds)

        grads = tape.gradient(uq_loss, self.model.trainable_variables)

        # gradient clipping
        if self.config.max_grad_norm > 0:
            grads, self._grad_norm = tf.clip_by_global_norm(grads, self.config.max_grad_norm)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return uq_loss_last_tar.numpy(), mse.numpy()

    def _write_train_logs(self, train_logs, name):
        """
        Writes training logs into a csv file
        :param train_logs: training logs dict
        :param name: name of log - _batch, _epoch
        :return:
        """
        df = pd.DataFrame.from_dict(train_logs)
        fname = self.config.name + '-' + name + '.csv'
        fname = os.path.join(self.train_log_dir, fname)
        df.to_csv(fname, sep=',', index=False)
        return

    def _save_criteria(self, train_logs_epoch):
        """
        model saving criteria where min valid mse model is saved
        :param train_logs_epoch:
        :return:
        """
        assert len(train_logs_epoch['valid_mse']) > 0, 'Error in computing valid_mse or incorrect train log dict passed'
        if train_logs_epoch['valid_mse'][-1] < self.min_valid_mse:
            self.min_valid_mse = train_logs_epoch['valid_mse'][-1]
            return True
        else:
            return False

    def _stop_criteria(self, train_logs_epoch):
        """
        Criteria to stop training before hitting max epoch. If `valid_mse` does not improve for consecutive
        `early_stop` epochs, training is stopped
        :param train_logs_epoch:
        :return:
        """
        assert len(train_logs_epoch['valid_mse']) > 0, \
            'Error in computing valid_mse or incorrect train log dict passed'
        valid_mse = np.array(train_logs_epoch['valid_mse'])
        if valid_mse.shape[0] - np.argmin(valid_mse) + 1 >= self.config.early_stop:
            return True
        else:
            return False

    def _validation_metrics_point_estimate(self):
        """
        Returns the loss, accuracy metrics on the validation set using model's current parameters for point estimate
        models
        :return:
        TODO: Add other validation metrics such as accuracy, forecast error, etc. Only mse is implemented at present.
        """
        preds, targets = [], []
        preds_unscaled, targets_unscaled = [], []

        # for (batch_n, valid_set_items) in enumerate(self.valid_set):
        #     inp_idxs = valid_set_items[0]
        #     tar_idxs = valid_set_items[1]
        #     pre_metadata = valid_set_items[2]
        #     inp, batch_target, metadata = self.dataset.get_batch(inp_idxs, tar_idxs, pre_metadata)

        for (batch_n, cur_batch) in enumerate(self._valid_batches):
            inp = cur_batch[0]
            batch_target = cur_batch[1]
            metadata = cur_batch[2]

            batch_pred = self.model.predict(inp)

            # preds, targets need to be a list
            if self.config.forecast_steps == 1:
                batch_pred = [batch_pred]
                batch_target = [batch_target]

            assert isinstance(batch_pred, (list, tuple))
            assert isinstance(batch_target, (list, tuple))

            assert len(batch_pred) == len(batch_target)

            batch_pred_unscaled, batch_target_unscaled = copy.deepcopy(batch_pred), copy.deepcopy(batch_target)

            # unscale preds and targets
            for i in range(len(batch_pred_unscaled)):
                batch_pred_unscaled[i] = self._unscale_preds(batch_pred_unscaled[i])
                batch_target_unscaled[i] = self._unscale_preds(batch_target_unscaled[i])

            preds.append(batch_pred)
            targets.append(batch_target)
            preds_unscaled.append(batch_pred_unscaled)
            targets_unscaled.append(batch_target_unscaled)

        preds_all_batches, targets_all_batches = [], []
        preds_unscaled_all_batches, targets_unscaled_all_batches = [], []

        for i in range(self.config.forecast_steps):
            # scaled
            pred_i = np.vstack([x[i] for x in preds])
            target_i = np.vstack([x[i] for x in targets])
            preds_all_batches.append(pred_i)
            targets_all_batches.append(target_i)

            # unscaled
            pred_unscaled_i = np.vstack([x[i] for x in preds_unscaled])
            target_unscaled_i = np.vstack([x[i] for x in targets_unscaled])
            preds_unscaled_all_batches.append(pred_unscaled_i)
            targets_unscaled_all_batches.append(target_unscaled_i)

        _, valid_mse = self.losses.weight_adjusted_mse(targets_all_batches, preds_all_batches, True)

        # valid_mse_fcst is normalized by seq_norm such as mrkcap
        _, valid_mse_fcst = self.losses.weight_adjusted_mse(targets_unscaled_all_batches,
                                                            preds_unscaled_all_batches,
                                                            True)
        valid_uq_loss = None
        return valid_uq_loss, valid_mse.numpy(), valid_mse_fcst.numpy()

    def _validation_metrics_uq_range_estimate(self):
        """
        Returns the loss, accuracy metrics on the validation set using model's current parameters for uq range estimate
        models
        :return:
        TODO: Add other validation metrics such as accuracy, forecast error, etc. Only mse is implemented at present.
        """
        preds, var, targets = [], [], []
        preds_unscaled, targets_unscaled = [], []

        # for (batch_n, valid_set_items) in enumerate(self.valid_set):
        #     inp_idxs = valid_set_items[0]
        #     tar_idxs = valid_set_items[1]
        #     pre_metadata = valid_set_items[2]
        #     inp, batch_target, metadata = self.dataset.get_batch(inp_idxs, tar_idxs, pre_metadata)

        for (batch_n, cur_batch) in enumerate(self._valid_batches):
            inp = cur_batch[0]
            batch_target = cur_batch[1]
            metadata = cur_batch[2]

            batch_pred = self.model.predict(inp)

            # targets need to be a list
            if self.config.forecast_steps == 1:
                batch_target = [batch_target]

            batch_tar_pred = batch_pred[0::2]  # target preds
            batch_tar_var = batch_pred[1::2]  # variance preds

            assert isinstance(batch_tar_pred, (list, tuple))
            assert isinstance(batch_tar_var, (list, tuple))
            assert isinstance(batch_target, (list, tuple))

            assert len(batch_tar_pred) == len(batch_target)

            batch_tar_pred_unscaled, batch_target_unscaled = copy.deepcopy(batch_tar_pred), copy.deepcopy(batch_target)

            # unscale preds amd targets
            for i in range(len(batch_tar_pred_unscaled)):
                batch_tar_pred_unscaled[i] = self._unscale_preds(batch_tar_pred_unscaled[i])
                batch_target_unscaled[i] = self._unscale_preds(batch_target_unscaled[i])

            preds.append(batch_tar_pred)
            var.append(batch_tar_var)
            targets.append(batch_target)

            preds_unscaled.append(batch_tar_pred_unscaled)
            targets_unscaled.append(batch_target_unscaled)

        preds_all_batches, var_all_batches, targets_all_batches = [], [], []
        preds_unscaled_all_batches, targets_unscaled_all_batches = [], []

        for i in range(self.config.forecast_steps):
            # scaled
            pred_i = np.vstack([x[i] for x in preds]).astype('float32')
            var_i = np.vstack([x[i] for x in var]).astype('float32')
            target_i = np.vstack([x[i] for x in targets]).astype('float32')

            preds_all_batches.append(pred_i)
            var_all_batches.append(var_i)
            targets_all_batches.append(target_i)

            # unscaled
            pred_unscaled_i = np.vstack([x[i] for x in preds_unscaled])
            target_unscaled_i = np.vstack([x[i] for x in targets_unscaled])

            preds_unscaled_all_batches.append(pred_unscaled_i)
            targets_unscaled_all_batches.append(target_unscaled_i)


        # UQ Loss
        _, valid_uq_loss, valid_mse = self.losses.weight_adjusted_uq_loss(targets_all_batches,
                                                                          preds_all_batches,
                                                                          var_all_batches)

        # valid_mse_fcst is normalized by seq_norm such as mrkcap
        _, valid_mse_fcst = self.losses.weight_adjusted_mse(targets_unscaled_all_batches,
                                                            preds_unscaled_all_batches)

        return valid_uq_loss.numpy(), valid_mse.numpy(), valid_mse_fcst.numpy()

    def _unscale_preds(self, arr):
        """
        Unscales the input array by reversing parameter scaling and log squashing.
        Returns raw preds/targets which are normalized by seq_norm
        :param arr:
        :return: raw preds, targets normalized by seq_norm
        """
        arr = np.multiply(arr, self.dataset.scaling_params['scale'][:self.dataset.n_outputs]) + \
              self.dataset.scaling_params['center'][:self.dataset.n_outputs]

        if self.config.log_squasher:
            arr = self.dataset.reverse_log_squasher(arr)
        return arr


if __name__ == '__main__':
    config = get_configs()
    config.train = True
    config.scale_field = 'mrkcap'
    config.log_squasher = True
    config.max_unrollings = 5
    config.forecast_n = 12
    config.forecast_steps = 1
    config.start_date = 197501
    config.end_date = 199912
    # config.batch_size = 1
    config.lr_decay = 1.0
    config.dropout = 0.3
    config.recurrent_dropout = 0.3
    config.nn_type = 'RNNPointEstimate'
    config.UQ = False
    config.max_epoch = 5
    config.early_stop = 10

    # config.datafile = "source-ml-data-v8-100M.dat"
    config.datafile = "sample_data_testing.dat"
    config.data_dir = "../datasets"
    config.model_dir = "../experiments/test-model"
    config.validation_size = 0.3
    D = Dataset(config)
    # print(D._train_gvkeys, D._valid_gvkeys, D._test_gvkeys)

    # rnn_model = RNNPointEstimate(config, D)
    # m = rnn_model.model
    # print(m.summary())
    # train = Train(config, m, D)
    train = Train(config, D)
    train.train()
