from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import platform
import os
import time
import numpy as np
import pandas as pd
# import matplotlb.pyplot as plt
import tensorflow as tf
from collections import defaultdict
import pathlib
from model_utils.model import Model
# from models.point_estimate.rnn_point_estimate import RNNPointEstimate
from base_config import get_configs
from data_processing import CDRSInferenceData
from data_processing import Dataset


physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

np.set_printoptions(threshold=sys.maxsize)


class Predict(object):

    def __init__(self, config, dataset):

        self.config = config
        assert not self.config.train, 'Predict can only be instantiated when config.train is False'
        self.dataset = dataset
        self.dataset.generate_dataset()
        self.model = Model(self.config, self.dataset).get_model()
        self.target_index = self.dataset.target_index
        self.seq_len = self.dataset.seq_len
        self.n_inputs = self.dataset.n_inputs

        # Load data
        self.test_set = self.dataset.test_set

        self.test_set = self.test_set.batch(batch_size=self.config.batch_size)

        # Create model level directories for chkpts and logs
        self.model_dir = os.path.join(self.config.experiments_dir, self.config.model_dir)
        self.chkpts_dir = os.path.join(self.model_dir, 'chkpts')

        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)

        print('creating preds directory')
        self.pred_dir = os.path.join(self.model_dir, 'pred')

        if 'Naive' not in self.config.nn_type:
            assert os.path.isdir(self.chkpts_dir), 'No checkpoint dir found to load the model'

        if not os.path.isdir(self.pred_dir):
            pathlib.Path(self.pred_dir).mkdir(parents=True, exist_ok=True)

        # Create Batches
        print("Creating batches ...")
        self._batches = []
        for (batch_n, train_set_items) in enumerate(self.test_set):
            inp_idxs = train_set_items[0]
            tar_idxs = train_set_items[1]
            pre_metadata = train_set_items[2]
            inp, target, metadata = self.dataset.get_batch(inp_idxs, tar_idxs, pre_metadata)
            self._batches.append((inp, target, metadata))

    def predict(self):
        """
        for each batch
            preds, metadata(targets, dates, etc) <- model(inp)

        combine all batches for preds, metadata

        normalized_targets <- targets/seq_norm
        compute uq_loss, mse
        compute forecast error, std dev, frequency etc

        save dataframes for lower bounds, pred, targets for each forecast step
            df: date, gvkey, preds, targets, LB_preds, abs error

        :return:

        """
        # Load model
        checkpoint_path = os.path.join(self.chkpts_dir, "chkpt")
        if 'Naive' in self.config.nn_type:
            pass
        else:
            self.model.load_weights(checkpoint_path)

        outputs = defaultdict(list)

        # for (batch_n, test_set_items) in enumerate(self.test_set):
        #     inp_idxs = test_set_items[0]
        #     tar_idxs = test_set_items[1]
        #     pre_metadata = test_set_items[-1]
        #     inp, targets, metadata = self.dataset.get_batch(inp_idxs, tar_idxs, pre_metadata)
        #     inp, targets = inp.numpy(), [targets.numpy()]
            # print(batch_n, metadata)
            # print(inp)

        for (batch_n, cur_batch) in enumerate(self._batches):
            inp = cur_batch[0]
            target = cur_batch[1]
            metadata = cur_batch[2]

            inp, targets = inp.numpy(), [target.numpy()]

            outputs['date'].append(np.expand_dims(metadata[:, 0].astype('int32'), -1))
            outputs['gvkey'].append(np.expand_dims(metadata[:, 1], -1))
            outputs['seq_norm'].append(np.expand_dims(metadata[:, 2].astype('float32'), -1))

            if self.config.write_inp_to_out_file:
                # extract inputs for each time step
                var_name = 'inp_t'
                for i, t_step in enumerate(range(1 - self.seq_len, 1)):
                    inp_t = self._extract_inputs(inp, i)
                    outputs[var_name + str(t_step)].append(inp_t)

            # extract the target id output for the last time step
            targets = self._extract_targets(targets)
            outputs['targets'].append(targets)

            if not self.config.UQ:
                preds = self.model.predict(inp)
                if self.config.forecast_steps == 1:
                    preds = [preds]
                # variances are zero for Point Estimates
                variance = [np.zeros(x.shape) for x in preds]

            else:  # UQ Range Estimate
                model_preds = self.model.predict(inp)
                preds = model_preds[0::2]
                variance = model_preds[1::2]

            assert isinstance(preds, (list, tuple)), 'predictions should be a list of preds for all targets'
            assert isinstance(variance, (list, tuple)), 'variances should be a list of preds for all targets'

            preds = self._extract_preds(preds)
            variance = self._extract_preds(variance)
            outputs['norm_preds'].append(preds)
            outputs['norm_variance'].append(variance)

        single_outputs = ['date', 'gvkey', 'seq_norm'] + [x for x in outputs.keys() if 'inp_t' in x]
        multiple_ouptuts = ['targets', 'norm_preds', 'norm_variance']

        for key in single_outputs:
            outputs[key] = np.vstack(outputs[key])

        for key in multiple_ouptuts:
            for i in range(self.config.forecast_steps):
                name = key + '_' + str(i + 1)
                outputs[name] = np.vstack([x[i] for x in outputs[key]])

        # pop unused keys. Only keep targets_1, norm_preds_1, ...
        outputs.pop('targets')
        outputs.pop('norm_preds')
        outputs.pop('norm_variance')

        outputs = {k: v.flatten() for k, v in outputs.items()}

        # create output dataframe
        df = pd.DataFrame.from_dict(outputs)
        try:
            df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
        except ValueError:
            print("Input date is not in the '%Y%m%d' format")
            raise

        tar_cols = [x for x in df.columns if 'targets' in x]
        norm_tar_cols = ['norm_' + x for x in tar_cols]

        # create norm_target copies before reversing targets and norm_squared_diff
        for norm_tar_c, tar_c in zip(norm_tar_cols, tar_cols):
            step = tar_c.split('_')[-1]
            norm_pred_c = 'norm_preds_' + step
            df[norm_tar_c] = df[tar_c]
            df['norm_squared_diff_' + step] = (df[norm_tar_c] - df[norm_pred_c]).apply(np.square)

        # reverse transform targets
        # reverse center and scale
        df.loc[:, tar_cols] = np.multiply(df.loc[:, tar_cols],
                                          self.dataset.scaling_params['scale'][self.target_index]) + \
                              self.dataset.scaling_params['center'][self.target_index]

        # reverse log squasher
        if self.config.log_squasher:
            df.loc[:, tar_cols] = self.dataset.reverse_log_squasher(df.loc[:, tar_cols])

        # un-normalize seq_norm
        df.loc[:, tar_cols] = df.loc[:, tar_cols].multiply(df['seq_norm'], axis='index')

        # reverse transform input time step values for target field
        if self.config.write_inp_to_out_file:
            inp_t_cols = [x for x in df.columns if 'inp_t' in x]

            # reverse center and scale
            df.loc[:, inp_t_cols] = np.multiply(df.loc[:, inp_t_cols],
                                                self.dataset.scaling_params['scale'][self.target_index]) + \
                                    self.dataset.scaling_params['center'][self.target_index]

            # reverse log squasher
            if self.config.log_squasher:
                df.loc[:, inp_t_cols] = self.dataset.reverse_log_squasher(df.loc[:, inp_t_cols])

            # un-normalize seq norm
            df[inp_t_cols] = df[inp_t_cols].multiply(df['seq_norm'], axis='index')

        # reverse transform to get raw preds
        preds_keys = [x for x in df.columns if 'norm_preds_' in x]
        for pred_key in preds_keys:
            pred_col_name = 'preds_' + pred_key.split('_')[-1]
            df[pred_col_name] = df[pred_key]

            # reverse center and scale using scaling params
            df.loc[:, pred_col_name] = np.multiply(df.loc[:, pred_col_name],
                                                   self.dataset.scaling_params['scale'][self.target_index]) + \
                                       self.dataset.scaling_params['center'][self.target_index]

            # Reverse log squasher
            if self.config.log_squasher:
                df.loc[:, pred_col_name] = self.dataset.reverse_log_squasher(df.loc[:, pred_col_name])

            # un-normalize by seq_norm
            df[pred_col_name] = df[pred_col_name].multiply(df['seq_norm'], axis='index')

        # reverse transform to get raw variance
        var_keys = [x for x in df.columns if 'norm_variance_' in x]
        for var_key in var_keys:
            var_col_name = 'variance_' + var_key.split('_')[-1]
            df[var_col_name] = df[var_key]

            # reverse center and scale using scaling params
            df.loc[:, var_col_name] = np.multiply(df.loc[:, var_col_name],
                                                  self.dataset.scaling_params['scale'][self.target_index]) + \
                                      self.dataset.scaling_params['center'][self.target_index]

            # Reverse log squasher
            if self.config.log_squasher:
                df.loc[:, var_col_name] = self.dataset.reverse_log_squasher(df.loc[:, var_col_name])

            # un-normalize by seq_norm
            df[var_col_name] = df[var_col_name].multiply(df['seq_norm'], axis='index')

        # forecast error
        output_keys = [x.split('_')[-1] for x in preds_keys]
        for o_key in output_keys:
            fcst_err_name = 'fcst_err_' + o_key
            df[fcst_err_name] = ((df['targets_' + o_key] - df['preds_' + o_key]) / df['targets_' + o_key]).abs()

            abs_err_name = 'abs_err_' + o_key
            df[abs_err_name] = (df['targets_' + o_key] - df['preds_' + o_key]).abs()

            se_name = 'unscaled_squared_err_' + o_key
            df[se_name] = (df[abs_err_name] / df['seq_norm']).apply(np.square)

        # convert field types
        encoding = 'utf-8'
        df['gvkey'] = df['gvkey'].apply(lambda x: x.decode(encoding))
        df.to_csv(os.path.join(self.pred_dir, self.config.preds_fname), sep=' ', index=False, date_format="%Y%m%d")

        print("Unscaled MSE normalized by Seq_Norm (%s)" % self.config.model_dir.split('/')[-1])
        unscaled_cols = [x for x in df.columns if 'unscaled' in x]
        print(df[unscaled_cols].mean())

        for i in range(self.config.forecast_steps):
            print("Scaled MSE for step %i: %1.4f" % (i + 1, df['norm_squared_diff_' + str(i + 1)].mean()))

        return df

    def _extract_targets(self, targets):
        if 'RNN' in self.config.nn_type:
            targets = [np.expand_dims(x[:, -1, self.target_index], -1) for x in targets]
        elif 'MLP' in self.config.nn_type or 'Naive' in self.config.nn_type:
            targets = [np.expand_dims(x[:, self.target_index], -1) for x in targets]
        else:
            raise NotImplementedError
        return targets

    def _extract_preds(self, preds):
        if 'RNN' in self.config.nn_type:
            preds = [np.expand_dims(x[:, -1, self.target_index], -1) for x in preds]
        elif 'MLP' in self.config.nn_type or 'Naive' in self.config.nn_type:
            preds = [np.expand_dims(x[:, self.target_index], -1) for x in preds]
        else:
            raise NotImplementedError
        return preds

    def _extract_inputs(self, inp, i):
        if 'RNN' in self.config.nn_type:
            inp_t = np.expand_dims(inp[:, i, self.target_index], -1)
        elif 'MLP' in self.config.nn_type or 'Naive' in self.config.nn_type:
            inp_t = np.expand_dims(inp[:, (i * self.n_inputs) + self.target_index], -1)
        else:
            raise NotImplementedError
        return inp_t


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 20)

    config = get_configs()
    config.train = False
    config.scale_field = 'mrkcap'
    config.max_unrollings = 5
    config.forecast_n = 12
    config.forecast_steps = 1
    config.start_date = 200001
    config.end_date = 201012
    # config.batch_size = 1
    config.lr_decay = 1.0
    config.dropout = 0.3
    config.recurrent_dropout = 0.3
    config.nn_type = 'RNNPointEstimate'
    config.UQ = False

    # config.datafile = "source-ml-data-v8-100M.dat"
    config.datafile = "sample_data_testing.dat"
    # config.datafile = 'cdrs-ml-data-2020-03-03.dat'
    config.data_dir = "../datasets"
    config.model_dir = "../experiments/test-model"
    D = Dataset(config)

    # D._test_gvkeys = ['100401', '101301']
    # print(D._train_gvkeys, D._valid_gvkeys, D._test_gvkeys)

    # rnn_model = RNNPointEstimate(config, D)
    # m = rnn_model.model
    # print(m.summary())
    # pred = Predict(config, m, D)
    pred = Predict(config, D)
    pred.predict()
