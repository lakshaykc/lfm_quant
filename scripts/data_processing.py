from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import copy
from datetime import datetime
import time
import pickle
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import pathlib
from sklearn import preprocessing as sk_pre

from base_config import get_configs

_MIN_SEQ_NORM = 10


class Dataset(object):
    """
    Builds training, validation and test datasets based on ```tf.data.Dataset``` type

    Attributes:

    Methods:
    """

    def __init__(self, config):
        self.config = config
        self._data_path = os.path.join(self.config.data_dir, self.config.datafile)

        self.is_train = self.config.train
        self.seq_len = self.config.max_unrollings

        # read and filter data_values based on start and end date
        self.data = pd.read_csv(self._data_path, sep=' ', dtype={'gvkey': str})
        try:
            self.data['date'] = pd.to_datetime(self.data['date'], format="%Y%m%d")
            self.start_date = pd.to_datetime(self.config.start_date, format="%Y%m%d")
            self.end_date = pd.to_datetime(self.config.end_date, format="%Y%m%d")
        except ValueError:
            # time data does not match the format "%Y%m%d". For example non-cdrs data is in the format "%Y%m"
            self.data['date'] = pd.to_datetime(self.data['date'], format="%Y%m")
            self.start_date = pd.to_datetime(self.config.start_date, format="%Y%m")
            self.end_date = pd.to_datetime(self.config.end_date, format="%Y%m")

        # date_offset_from_end is added end date to ensure tha target date is not Nan if data_values is available
        self._date_offset_from_end = pd.DateOffset(months=self.config.stride)
        # date_offset_from_start is subtracted to start_date to ensure enough history is present to build the dataset
        # from the start_date
        self._date_offset_from_start = pd.DateOffset(years=self.config.max_unrollings)
        self.data = self.data[(self.data['date'] >= self.start_date - self._date_offset_from_start) & \
                                            (self.data['date'] <= self.end_date + self._date_offset_from_end)]

        # split gvkeys into train, validation
        self.gvkeys = self._get_gvkeys()
        self._train_gvkeys, self._valid_gvkeys, self._test_gvkeys = self.train_test_split(self.gvkeys,
                                                                                          self.config.validation_size,
                                                                                          self.config.seed,
                                                                                          self.is_train)

        print("Start Date: %s" % self.start_date.strftime('%Y-%m-%d'))
        print("End Date: %s" % self.end_date.strftime('%Y-%m-%d'))
        print("Loading dataset %s complete" % self.config.datafile)
        print("Total number of records: %i" % self.data.shape[0])
        if self.config.train:
            print("Run type: Training")
            print("Number of training entities: %i" % len(self._train_gvkeys))
            print("Number of validation entities: %i" % len(self._valid_gvkeys))
        else:
            print("Run type: Prediction")
            print("Number of test entities: %i" % len(self._test_gvkeys))

        _, self.fin_col_names = self.get_cols_from_colnames(self.config.financial_fields)
        _, self.aux_col_names = self.get_cols_from_colnames(self.config.aux_fields)
        _, self.dont_scale_col_names = self.get_cols_from_colnames(self.config.dont_scale_fields)

        self.n_inputs = len(self.fin_col_names) + len(self.aux_col_names)
        self.n_outputs = len(self.fin_col_names)

        # target_index refers to the column index of the target variable in the target data_values. Target
        # data_values is essentially the input data_values for fin_col_names shifted by stride. This mean the row for
        # next time step is the target for current time step.
        self.target_index = self.fin_col_names.index(self.config.target_field)

        self._cols = ['date', 'gvkey', 'active'] + self.fin_col_names + self.aux_col_names
        self._cols_offset = 3  # ['date', 'gvkey', ''active]

        # Append scale field if not in fin_cols + aux_cols
        if self.config.scale_field in self.data.columns and self.config.scale_field not in self._cols:
            self._cols.append(self.config.scale_field)

        # self.data = self.data[['date', 'gvkey', 'active'] + self.fin_col_names + self.aux_col_names]
        self.data = self.data[self._cols]

        # self._cols = self.data.columns.tolist()
        self._gvkey_idx = self._cols.index(self.config.key_field)
        self._date_idx = self._cols.index(self.config.date_field)
        self._active_idx = self._cols.index(self.config.active_field)
        self.fin_col_ids = [self._cols.index(x) for x in self.fin_col_names]
        self.aux_col_ids = [self._cols.index(x) for x in self.aux_col_names]
        self.dont_scale_col_ids = [self._cols.index(x) for x in self.dont_scale_col_names]
        self.inp_col_ids = self.fin_col_ids + self.aux_col_ids

        # scale_inp_col_ids are indexes of input tensor and not the _data matrix. _data matrix contains additional
        # columns such as date, gvkey. input tensors only contains fin + aux fields.
        self.scale_inp_col_ids = [x - self._cols_offset for x in self.inp_col_ids if x not in self.dont_scale_col_ids]

        self._aux_col_ids_seq = [x - self._cols_offset for x in self.aux_col_ids]

        # sequence normalizer
        if self.config.scale_field in self._cols:
            self._seq_norm_idx = self._cols.index(self.config.scale_field)
        else:
            self._seq_norm_idx = None

        # convert data_values from dataframe to numpy array for faster ops
        self.data_values = self.data.values
        self._dataset = {'train_X': [],
                         'train_Y': [],
                         'valid_X': [],
                         'valid_Y': [],
                         'test_X': [],
                         'test_Y': [],
                         'pre_metadata': []
                         }
        # self._dataset values are converted from a list of sequences to np array of shape=(samples, seq_len, features)

        self.model_dir = None
        self.scaling_params = None

    def generate_dataset(self):
        """
        Generates the dataset by adding dataset properties such as train_set, test_set to the instantiated class
        Also ensures scaling params are loaded and/or saved
        :return:
        """
        # build tf.data_values.dataset
        self._create_tf_dataset()

        # scaling params
        self.model_dir = os.path.join(self.config.experiments_dir, self.config.model_dir)
        if not os.path.isdir(self.model_dir):
            pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        if not self.config.scalesfile:
            scales_path = os.path.join(self.model_dir, 'scales.dat')
        else:
            scales_path = self.config.scalesfile

        if self.config.train:
            try:
                self.scaling_params = pickle.load(open(scales_path, 'rb'))
            except FileNotFoundError:
                self.scaling_params = self.get_scaling_params()
                pickle.dump(self.scaling_params, open(scales_path, 'wb'))

        else:  # prediction
            assert os.path.isfile(scales_path), "scalesfile not provided. Ensure to use the same scalesfile as used " \
                                                "during training "
            self.scaling_params = pickle.load(open(scales_path, 'rb'))

        # print dataset stats
        # self._print_dataset_stats()

    def _create_tf_dataset(self):
        """
        Builds training set, validation set as tf.data.dataset objects if config.train=True.
        Otherwise builds testing set as tf.data.dataset object.

        For each gvkey, sequences of len `max_unrollings` are built where the last element of the sequence is active.
        Sequences are padded with zeros if enough historical data is not present to meet `max_unrollings`. For example,
        if start_date=1980, max_unrollings=8 and company x only has data beginning 1975, sequences from 1971 to 1975
        will be padded with zeros.

        If targets are not present for inputs, Nans are returned. For example, if training end date is today's date,
        target data for 1 year into the future is will not be available, hence Nans are returned.


        shuffling and batching should be performed during training/ prediction


        Pseudo Code
        dataset = []
        for i in len(data):

            if active and
            start_date < date < end_date and
            has enough history (seq_len):

                create the sequence as np.array
                pad with zeros if seq_len < max_unrollings
                append to corresponding dataset (train, valid or test)

        concat sequences (np.arrays) to create np.array of shape=(num_sequences, seq_len, num_inputs/ num_outputs)
        convert to tf.data.dataset object
        """

        min_steps = self.config.stride * (self.config.min_unrollings - 1) + 1
        max_steps = self.config.stride * (self.config.max_unrollings - 1) + 1

        last_key = ''
        cur_len = 1

        for i in range(self.data_values.shape[0]):
            key = self.data_values[i, self._gvkey_idx]
            active = True if int(self.data_values[i, self._active_idx]) else False
            date = self.data_values[i, self._date_idx]
            if i + self.config.forecast_n <= self.data_values.shape[0] - 1:
                tar_key = self.data_values[i + self.config.forecast_n, self._gvkey_idx]
            else:
                tar_key = ''

            if key != last_key:
                cur_len = 1
            if self.config.train:  # Training
                if cur_len >= min_steps \
                        and active is True \
                        and self.start_date <= date <= self.end_date - pd.DateOffset(months=self.config.stride) \
                        and tar_key == key:
                    self._append_sequence_data(cur_len, min_steps, max_steps, i, key, tar_key, date)

                cur_len += 1
                last_key = key

            else:  # Prediction
                if cur_len >= min_steps \
                        and active is True \
                        and self.start_date <= date <= self.end_date:
                    self._append_sequence_data(cur_len, min_steps, max_steps, i, key, tar_key, date)

                cur_len += 1
                last_key = key

        for k, v in self._dataset.items():
            if len(v) > 0:
                self._dataset[k] = np.concatenate(v, axis=0)
            else:
                self._dataset[k] = None

        return

    def _append_sequence_data(self, cur_len, min_steps, max_steps, idx, key, tar_key, date):
        """
        Appends two forms of data to their respective dataset attributes
        1. append [start_idc, end_idx, pad_size] for inputs, targets to self._dataset['*_X'], self._dataset['*_Y'], etc
        2. append pre_metadata [date, inp_key, tar_key] to self._dataset['pre_metadata']
        :param cur_len: len of current key's elements from it's start date
        :param min_steps: number of min steps to form a sequence
        :param max_steps: number of max steps to form a sequence
        :param idx: index of the last element in the sequence
        :param key: input gvkey
        :param tar_key: gvkey of target
        :param date: date of input gvkey
        :return:
        """
        assert cur_len >= min_steps

        seq_len = min(cur_len - (cur_len - 1) % self.config.stride, max_steps)
        pad_size = (max_steps - seq_len) // self.config.stride

        # train/valid/test sets define indices and are of the form [start_idx, end_idx, pad_size]
        inp_indices = np.expand_dims(np.array([idx - seq_len + 1, idx, pad_size]),
                                     axis=0)

        if key == tar_key:
            tar_indices = np.expand_dims(np.array([idx - seq_len + 1 + self.config.forecast_n,
                                                   idx + self.config.forecast_n,
                                                   pad_size]),
                                         axis=0)
        else:  # tar_key could be another gvkey or '' i.e end of datafile
            tar_indices = np.expand_dims(np.array([idx - seq_len + 1 + self.config.forecast_n,
                                                   idx,
                                                   pad_size]),
                                         axis=0)

        pre_metadata = np.expand_dims(np.array([date.strftime("%Y%m%d"), key, tar_key]),
                                      axis=0)

        self._append_idxs_to_dataset(key, inp_indices, tar_indices)
        self._append_pre_metadata_to_dataset(pre_metadata)

    def _append_idxs_to_dataset(self, key, inp_indices, tar_indices):
        if key in self._train_gvkeys and self.config.train:
            self._dataset['train_X'].append(inp_indices)
            self._dataset['train_Y'].append(tar_indices)
        elif key in self._valid_gvkeys and self.config.train:
            self._dataset['valid_X'].append(inp_indices)
            self._dataset['valid_Y'].append(tar_indices)
        elif key in self._test_gvkeys and not self.config.train:
            self._dataset['test_X'].append(inp_indices)
            self._dataset['test_Y'].append(tar_indices)
        else:
            raise ValueError("Mismatch between gvkey category (train/valid/test set) and run type (train/ pred)")

    def _append_pre_metadata_to_dataset(self, pre_metadata):
        """
        appends pre_metadata of the form np.array([[date, key]]) to the dataset
        :return:
        """
        self._dataset['pre_metadata'].append(pre_metadata)

    def get_batch(self, inp_indices, tar_indices, pre_metadata):
        """
        creates a batch of inps, tar given the corresponding start indices, end indices and padding size.
        :param inp_indices: batch of [start_idxs, end_idxs, pad_size] for inputs. shape=(batch_size, 3)
        :param tar_indices: batch of [start_idxs, end_idxs, pad_size] for targets. shape=(batch_size, 3)
        :param pre_metadata: batch of metadata [date, gvkey, seq_norm]. shape=(batch_size, 3).
        :return: inp_batch, target_batch, metadata_batch.

        metadata_batch is None during training and is only stored during prediction

        batch shape: (batch_size, seq_len, features)
        """
        inp_indices, tar_indices, pre_metadata = inp_indices.numpy(), tar_indices.numpy(), pre_metadata.numpy()

        inp_batch = np.empty(shape=(inp_indices.shape[0], self.seq_len, len(self.inp_col_ids)))
        tar_batch = np.empty(shape=(tar_indices.shape[0], self.seq_len, self.n_outputs))
        metadata_batch = pre_metadata

        for i in range(inp_indices.shape[0]):
            inp_key, tar_key = pre_metadata[i, 1], pre_metadata[i, 2]
            if self.config.train:
                inp_batch[i, :, :], seq_norm = self._get_train_seq(inp_indices[i][0], inp_indices[i][1],
                                                                   inp_indices[i][2],
                                                                   self.inp_col_ids)
                tar_batch[i, :, :], _ = self._get_train_seq(tar_indices[i][0], tar_indices[i][1], tar_indices[i][2],
                                                            self.fin_col_ids)
            else:
                inp_batch[i, :, :], seq_norm = self._get_pred_seq(inp_indices[i][0], inp_indices[i][1],
                                                                  inp_indices[i][2],
                                                                  inp_key, tar_key, self.inp_col_ids)
                tar_batch[i, :, :], _ = self._get_pred_seq(tar_indices[i][0], tar_indices[i][1], tar_indices[i][2],
                                                           inp_key, tar_key, self.fin_col_ids)

            # Sequence Normalization
            inp_batch[i, :, 0:len(self.fin_col_ids)] /= seq_norm
            tar_batch[i, :, 0:len(self.fin_col_ids)] /= seq_norm
            # Log squasher
            if self.config.log_squasher:
                inp_batch[i, :, 0:len(self.fin_col_ids)] = self.log_squasher(inp_batch[i, :, 0:len(self.fin_col_ids)])
                tar_batch[i, :, 0:len(self.fin_col_ids)] = self.log_squasher(tar_batch[i, :, 0:len(self.fin_col_ids)])

            # overwrite tar_key column with seq_norm in pre_metadata to form metadata_batch
            metadata_batch[i, 2] = seq_norm

        # scaling params
        inp_batch[:, :, self.scale_inp_col_ids] = np.divide(
            inp_batch[:, :, self.scale_inp_col_ids] - \
            self.scaling_params['center'][self.scale_inp_col_ids],
            self.scaling_params['scale'][self.scale_inp_col_ids])
        tar_batch = np.divide(tar_batch - self.scaling_params['center'][:len(self.fin_col_ids)],
                              self.scaling_params['scale'][:len(self.fin_col_ids)])

        if self.config.aux_masking:
            # make aux fields to 0 for all time steps except the last one
            inp_batch[:, 0:self.seq_len - 1, self._aux_col_ids_seq] = 0.0

        if 'MLP' in self.config.nn_type or 'Naive' in self.config.nn_type:
            inp_batch = inp_batch.reshape(inp_indices.shape[0], self.seq_len * len(self.inp_col_ids))
            tar_batch = tar_batch[:, -1, :]

        return tf.convert_to_tensor(inp_batch, dtype=tf.float32), tf.convert_to_tensor(tar_batch, dtype=tf.float32), \
               metadata_batch

    def _get_train_seq(self, start_idx, end_idx, pad_size, col_ids):
        """
        returns sequence for training of dtype np.ndarray and shape=(seq_len, len(col_ids). The first pad_size elements
        in the sequence are zero is start_idx and end_idx do not form a sequence of seq_len.

        end_idx should exist in the data as targets cannot be nan during training

        :param start_idx: starting index of the sequence
        :param end_idx: end index of the sequence
        :param pad_size: number of elements in the sequence to be padded to match seq_len
        :param col_ids: column ids to create the sequences. Example input/ output column ids
        :return: sequence dtype: np.ndarray, shape=(seq_len, len(col_ids), seq_norm
        """
        if pad_size > 0:
            seq = np.concatenate([np.zeros(shape=(pad_size, self.data_values.shape[1])),
                                  self.data_values[
                                  start_idx: end_idx + self.config.stride: self.config.stride,
                                  :]],
                                 axis=0)
        else:
            seq = self.data_values[start_idx: end_idx + self.config.stride: self.config.stride, :]

        # Sequence Normalization
        if self._seq_norm_idx:
            seq_norm = max(seq[-1, self._seq_norm_idx], _MIN_SEQ_NORM)
        else:
            seq_norm = 1.

        return seq[:, col_ids], seq_norm

    def _get_pred_seq(self, start_idx, end_idx, pad_size, inp_key, tar_key, col_ids):
        """
        returns sequence for prediction of dtype np.ndarray and shape=(seq_len, len(col_ds))
        If start_idx and end_idx do not form a sequence of seq_len, zero padding is used.

        if end_idx > len(data) or gvkey of end_idx is not the same as current gvkey:
            Nans are returned

        :param start_idx: starting index of the sequence
        :param end_idx: end index of the sequence
        :param pad_size: umber of elements in the sequence to be padded to match seq_le
        :param inp_key: gvkey of input sequence
        :param tar_key: gvkey of target sequence
        :param col_ids: column ids to create the sequences. Example input/ output column ids
        :return: sequence dtype: np.ndarray, shape=(seq_len, len(col_ids), seq_norm
        """

        if tar_key == inp_key:
            if pad_size > 0:
                seq = np.concatenate([np.zeros(shape=(pad_size, self.data_values.shape[1])),
                                      self.data_values[
                                      start_idx: end_idx + self.config.stride: self.config.stride,
                                      :]],
                                     axis=0)
            else:
                seq = self.data_values[start_idx: end_idx + self.config.stride: self.config.stride, :]

        else:  # target values don't exist for the given time period. Return Nan filled np array
            seq = np.empty((self.seq_len - pad_size, self.data_values.shape[1]))
            seq[:] = np.nan
            for i, j in enumerate(range(start_idx, end_idx + self.config.stride, self.config.stride)):
                try:
                    self.data_values[j, self._date_idx] = self.data_values[j, self._date_idx].strftime("%Y%m%d")
                except AttributeError:
                    pass
                seq[i, :] = self.data_values[j, :]

            if pad_size > 0:
                seq = np.concatenate([np.zeros(shape=(pad_size, seq.shape[1])), seq],
                                     axis=0)
            else:
                seq = seq

        # Sequence Normalization
        if self._seq_norm_idx:
            seq_norm = max(seq[-1, self._seq_norm_idx], _MIN_SEQ_NORM)
        else:
            seq_norm = 1.

        return seq[:, col_ids], seq_norm

    @property
    def train_set(self):
        assert self.config.train, 'config.train is not True. train_set is only available during training'
        X = tf.data.Dataset.from_tensor_slices(self._dataset['train_X'].astype('int32'))
        Y = tf.data.Dataset.from_tensor_slices(self._dataset['train_Y'].astype('int32'))
        return tf.data.Dataset.zip((X, Y, self.pre_metadata))

    @property
    def valid_set(self):
        assert self.config.train, "config.train is not True. valid_set is only available during training"
        X = tf.data.Dataset.from_tensor_slices(self._dataset['valid_X'].astype('int32'))
        Y = tf.data.Dataset.from_tensor_slices(self._dataset['valid_Y'].astype('int32'))
        return tf.data.Dataset.zip((X, Y, self.pre_metadata))

    @property
    def test_set(self):
        assert not self.config.train, "config.train is not False. test_set is only available during prediction"
        X = tf.data.Dataset.from_tensor_slices(self._dataset['test_X'].astype('int32'))
        Y = tf.data.Dataset.from_tensor_slices(self._dataset['test_Y'].astype('int32'))
        return tf.data.Dataset.zip((X, Y, self.pre_metadata))

    @property
    def pre_metadata(self):
        return tf.data.Dataset.from_tensor_slices(self._dataset['pre_metadata'])

    def get_cols_from_colnames(self, columns):
        """
        Returns indexes and names of columns of data that are included in columns.
        columns are separated by commas and can include ranges. For example,
        f1-f5,f7,f9 would be feature one through 5, and feature 7 and 9.

        :param columns: columns or column ranges from the config file. eg. f1-f5,f7,f8
        :return: column indices, column names
        """
        colidxs = []
        col_names = []

        if columns:
            data_cols = self.data.columns.tolist()
            col_list = columns.split(',')
            for col in col_list:
                col_range = col.split('-')
                if len(col_range) == 1:
                    colidxs.append(list(data_cols).index(col_range[0]))
                    col_names.append(col_range[0])
                elif len(col_range) == 2:
                    start_idx = list(data_cols).index(col_range[0])
                    end_idx = list(data_cols).index(col_range[1])
                    assert (start_idx >= 0)
                    assert (start_idx <= end_idx)
                    colidxs.extend(list(range(start_idx, end_idx + 1)))
                    col_names += [data_cols[i] for i in range(start_idx, end_idx + 1)]
        return colidxs, col_names

    def _get_gvkeys(self):
        """
        :return: all gvkeys between start and end date
        """
        assert isinstance(self.data, pd.DataFrame)
        gvkey_data = self.data[['date', 'gvkey']]
        gvkey_data = gvkey_data[gvkey_data['date'] <= self.end_date]

        return gvkey_data['gvkey'].unique()

    @staticmethod
    def train_test_split(keys, validation_size, seed, is_train):
        """
        Splits a list (eg gvkeys) into training, validation and test lists based on validation size as a fraction
        of list size.
        If `is_train=False` i.e prediction, training and validation set are empty. Tests set contains all the keys
        if `is_train=True`, test set is empty
        :param keys: list of gvkeys
        :param validation_size: fraction of total size of list to be used for validation
        :param seed: random seed for split
        :param is_train: split only for training. If is_train is False, training set is None
        :return: train_keys, valid_keys
        """
        np.random.seed(seed)
        if is_train:
            valid_keys = np.random.choice(keys, size=int(len(keys) * validation_size), replace=False)
            train_keys = list(set(keys) - set(valid_keys))
            test_keys = []
        else:
            train_keys = []
            valid_keys = []
            test_keys = keys
        return sorted(train_keys), sorted(valid_keys), sorted(test_keys)

    def get_scaling_params(self):
        """
        Returns scaling parameters of the sklearn scaler specified in the config file.
        :return: sklearn scaler
        """
        assert self.config.train, "scaling params are only calculated during training"

        indices_data = self._dataset['train_X'].tolist()

        data_sample = list()

        indices_sample = random.sample(indices_data, int(0.3 * len(indices_data)))

        for start_idx, end_idx, _ in indices_sample:
            step = random.randrange(self.config.min_unrollings)
            cur_idx = start_idx + step * self.config.stride
            assert cur_idx <= self.data_values.shape[0]
            x1 = self.get_feature_vector(cur_idx, end_idx)
            x2 = self.get_aux_vector(cur_idx)
            data_sample.append(np.append(x1, x2))

        scaler_class = self.config.data_scaler
        if hasattr(sk_pre, scaler_class):
            scaler = getattr(sk_pre, scaler_class)()
        else:
            raise RuntimeError("Unknown scaler = %s" % scaler_class)

        scaler.fit(data_sample)

        params = dict()
        params['center'] = scaler.center_ if hasattr(scaler, 'center_') else scaler.mean_
        params['scale'] = scaler.scale_

        return params

    def get_feature_vector(self, cur_idx, end_idx):
        """
        returns cur_idx feature vector normalized and log squashed by scale_field of last time step (i.e end_idx)
        :param cur_idx: idx of current time step
        :param end_idx: idx of last time step
        :return: np.array
        """
        x = self.data_values[cur_idx, self.fin_col_ids]
        if self._seq_norm_idx:
            normalizer = max(self.data_values[end_idx, self._seq_norm_idx], _MIN_SEQ_NORM)
        else:
            normalizer = 1.
        x = np.divide(x, normalizer)
        if self.config.log_squasher:
            x_abs = np.absolute(x).astype(float)
            x = np.multiply(np.sign(x), np.log1p(x_abs))
        return x

    def get_aux_vector(self, cur_idx):
        """
        return aux vector for cur_idx
        :param cur_idx:
        :return: np.array of shape=(len(aux_fields)
        """
        return self.data_values[cur_idx, self.aux_col_ids]

    def log_squasher(self, x):
        """
        Applies log squashing function i.e log(1 + x) if log_squasher is True in config
        :param x:
        :return:
        """
        if self.config.log_squasher:
            x_abs = np.absolute(x).astype(float)
            x = np.multiply(np.sign(x), np.log1p(x_abs))
        return x

    def reverse_log_squasher(self, x):
        """
        Reverses the log squashing function i. exp(x) - 1
        :param x:
        :return:
        """
        if self.config.log_squasher:
            x = np.multiply(np.sign(x), np.expm1(np.fabs(x)))
        return x

    def print_dataset_stats(self):
        print("[samples, sequence_length, features]")
        for k, v in self._dataset.items():
            if v is not None:
                print("%s : %s" % (k, v.shape,))
            else:
                print("%s: None" % k)


class CDRSInferenceData(Dataset):
    """
    Dataset class object to generate data_values from the CDRS dataset for inference
    """

    def __init__(self, config):
        super().__init__(config)

        assert self.end_date >= datetime.now(), """Ensure the end_date in the config is set to a value greater than
        today's date, preferably 220012"""

        assert self.start_date <= datetime.now() - pd.offsets.DateOffset(years=self.seq_len), """
        Ensure the start date goes back at least the seq_len, preferably 197501"""

        self.gvkeys = sorted(self.data['gvkey'].unique().tolist())
        self._train_gvkeys, self._valid_gvkeys, self._test_gvkeys = [], [], self.gvkeys

        assert not self.config.train, """CDRS data_values can only be used during inference. 
                    Ensure the train=False in the config"""

    def generate_dataset(self):
        """
        Generates dataset using CDRS Inference data_values as the source. Only builds the test_set as this is only applicable
        for inference. train, valid sets are empty lists.
        :return:
        """
        # build tf.data_values.dataset
        self._create_tf_dataset()

        # scaling params
        self.model_dir = os.path.join(self.config.experiments_dir, self.config.model_dir)
        if not self.config.scalesfile:
            scales_path = os.path.join(self.model_dir, 'scales.dat')
        else:
            scales_path = self.config.scalesfile
        assert os.path.isfile(scales_path), "scalesfile not provided. Make sure to use the same scalesfile as used " \
                                            "during training "
        self.scaling_params = pickle.load(open(scales_path, 'rb'))

        for k, v in self._dataset.items():
            if len(v) > 0:
                self._dataset[k] = np.concatenate(v, axis=0)
            else:
                self._dataset[k] = None

    def _create_tf_dataset(self):

        for i, gvkey in enumerate(self.gvkeys):
            gvkey_df = self.data[self.data.gvkey == gvkey]
            idxs = gvkey_df.index
            start_idx = idxs[0]
            end_idx = idxs[-1]
            pad_size = self.config.max_unrollings - len(idxs)
            date = self.data.loc[end_idx, 'date'].strftime("%Y%m%d")
            seq_norm = np.nan  # seq_norm is updated when batch is created using get_batch method

            self._dataset['test_X'].append(np.expand_dims(np.array([start_idx, end_idx, pad_size]),
                                                          axis=0))
            self._dataset['test_Y'].append(np.expand_dims(np.array([start_idx, end_idx, pad_size]),
                                                          axis=0))  # not used for predictions
            self._dataset['pre_metadata'].append(np.expand_dims(np.array([date, gvkey, seq_norm]),
                                                                axis=0))

    def get_batch(self, inp_indices, tar_indices, pre_metadata):
        """
        creates a batch of inps, tar given the corresponding start indices, end indices and padding size.
        :param inp_indices: batch of [start_idxs, end_idxs, pad_size] for inputs. shape=(batch_size, 3)
        :param tar_indices: batch of [start_idxs, end_idxs, pad_size] for targets. shape=(batch_size, 3)
        :param pre_metadata: batch of metadata [date, gvkey, seq_norm]. shape=(batch_size, 3).
        :return: inp_batch, target_batch, metadata_batch.

        metadata_batch is None during training and is only stored during prediction

        batch shape: (batch_size, seq_len, features)
        """
        inp_indices, tar_indices, pre_metadata = inp_indices.numpy(), tar_indices.numpy(), pre_metadata.numpy()

        inp_batch = np.empty(shape=(inp_indices.shape[0], self.seq_len, len(self.inp_col_ids)))
        tar_batch = np.empty(shape=(tar_indices.shape[0], self.seq_len, self.n_outputs))
        metadata_batch = pre_metadata

        # Note: Each gvkey has only one sequence
        for i in range(inp_indices.shape[0]):
            inp_batch[i, :, :], seq_norm = self.get_pred_seq(inp_indices[i][0],
                                                             inp_indices[i][1],
                                                             inp_indices[i][2],
                                                             self.inp_col_ids)
            # targets are not used when making predictions for the current date
            tar_batch[i, :, :] = np.nan

            # # Sequence normalization
            inp_batch[i, :, 0:len(self.fin_col_ids)] /= seq_norm
            # Log squasher
            if self.config.log_squasher:
                inp_batch[i, :, 0:len(self.fin_col_ids)] = self.log_squasher(inp_batch[i, :, 0:len(self.fin_col_ids)])

            # update seq_norm in metadata to form metadata_batch
            metadata_batch[i, 2] = seq_norm

        # scaling params
        inp_batch[:, :, self.scale_inp_col_ids] = np.divide(
            inp_batch[:, :, self.scale_inp_col_ids] - \
            self.scaling_params['center'][self.scale_inp_col_ids],
            self.scaling_params['scale'][self.scale_inp_col_ids])

        if self.config.aux_masking:
            # make aux fields to 0 for all time steps except the last one
            inp_batch[:, 0:self.seq_len - 1, self._aux_col_ids_seq] = 0.0

        if 'MLP' in self.config.nn_type or 'Naive' in self.config.nn_type:
            inp_batch = inp_batch.reshape(inp_indices.shape[0], self.seq_len * len(self.inp_col_ids))
            tar_batch = tar_batch[:, -1, :]

        return tf.convert_to_tensor(inp_batch, dtype=tf.float32), tf.convert_to_tensor(tar_batch, dtype=tf.float32), \
               metadata_batch

    def get_pred_seq(self, start_idx, end_idx, pad_size, col_ids):
        seq = self.data_values[start_idx: end_idx + 1, :]
        if pad_size > 0:
            seq = np.concatenate([np.zeros(shape=(pad_size, seq.shape[1])), seq],
                                 axis=0)
        else:
            seq = seq

        # Sequence normalization
        if self._seq_norm_idx:
            seq_norm = max(seq[-1, self._seq_norm_idx], _MIN_SEQ_NORM)
        else:
            seq_norm = 1.
        return seq[:, col_ids], seq_norm


if __name__ == '__main__':
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 100)
    # GCP Run
    t1 = time.time()
    config = get_configs()
    config.train = False
    # config.datafile = 'source-ml-data_values-v8-100M.dat'
    # config.datafile = 'sample_data_testing_4.dat'
    config.datafile = 'cdrs-ml-data.dat'
    config.data_dir = '../datasets'
    # config.model_dir = '../experiments/model'
    config.start_date = 20040101
    config.end_date = 22000101
    config.min_unrollings = 5
    config.max_unrollings = 5
    config.batch_size = 1
    config.scale_field = 'mrkcap'
    config.nn_type = 'RNNUqRangeEstimate'
    config.aux_masking = False
    config.model_dir = "../experiments/test-model"

    # D = Dataset(config)
    # D.generate_dataset()
    # print(D.scaling_params)
    # print(D._train_gvkeys, D._valid_gvkeys, D._test_gvkeys)
    
    # training_set = D.train_set.batch(batch_size=config.batch_size)
    # valid_set = D.valid_set.batch(batch_size=4)
    # pre_m_set = D.pre_metadata.batch(batch_size=4)

    # print(training_set)

    # for i, train_items in enumerate(training_set):
    #     pass
    # print(i)
    #     inp_idxs = train_items[0]
    #     tar_idxs = train_items[1]
    #     pre_m = train_items[2]
    #     inp, tar, metadata = D.get_batch(inp_idxs, tar_idxs, pre_m)
    #     if i == 0:
    #         print("INP")
    #         print(inp)
    #         print("TAR")
    #         print(tar)
    #         print("M")
    #         print(metadata)
    #     break

    # ------------------------------------------------------------------------
    # Test CDRSInferenceData
    cdrs_d = CDRSInferenceData(config)
    cdrs_d.generate_dataset()
    # # Load data_values
    # test_set = cdrs_d.test_set
    # test_set = test_set.batch(batch_size=config.batch_size)
    # # t = time.time()
    # for (batch_n, test_set_items) in enumerate(test_set):
    #     inp_idxs = test_set_items[0]
    #     tar_idxs = test_set_items[1]
    #     pre_metadata = test_set_items[-1]
    #     inp, targets, metadata = cdrs_d.get_batch(inp_idxs, tar_idxs, pre_metadata)
    #     # if batch_n % 100 == 0:
    #     # print(batch_n, time.time() - t)
    #     # t = time.time()
