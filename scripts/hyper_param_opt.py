#!/bin/sh
''''exec python3 -u -- "$0" ${1+"$@"} # '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import configs
from param_search.optimization import HPOptimization

_data_dir_path = os.environ.get('DEEP_QUANT_ROOT')
assert _data_dir_path is not None, "Environment Variable DEEP_QUANT_ROOT not set"


def get_configs():
    """
    Defines all configuration params passable to command line.
    """
    configs.DEFINE_string("name", 'hpo-test', "A name for the config.")
    configs.DEFINE_string("datafile", 'source-ml-data-v8-100M.dat', "a datafile name.")
    configs.DEFINE_string("predict_datafile", None,
                          "If predict_datafile is not None, use it instead of datafile for predictions")
    configs.DEFINE_string("mse_outfile", None, "A file to write mse values during predict phase.")
    configs.DEFINE_string("scalesfile", None, "Optional file for storing scaling params")
    configs.DEFINE_string("default_gpu", '/gpu:0', "The default GPU to use e.g., /gpu:0")
    configs.DEFINE_string("nn_type", 'RNNPointEstimate', "Model type")
    configs.DEFINE_string("active_field", 'active', "Key column name header for active indicator")
    configs.DEFINE_string("date_field", 'date', "Name of data column.")
    configs.DEFINE_string("key_field", 'gvkey', "Key column name header in datafile")
    configs.DEFINE_string("target_field", 'oiadpq_ttm', "Target column name header in datafile")
    configs.DEFINE_string("scale_field", 'mrkcap', "Feature to scale inputs by")
    configs.DEFINE_string("financial_fields", 'saleq_ttm-ltq_mrq', "Shared input and target field names")
    configs.DEFINE_string("aux_fields", 'rel_mom1m-rel_mom9m', "non-target, input only fields")
    configs.DEFINE_string("dont_scale", None, "Names of fields to not scale")
    configs.DEFINE_string("data_dir", 'datasets', "The data directory")
    configs.DEFINE_string("model_dir", 'test-model', "Model directory")
    configs.DEFINE_string("experiments_dir", './', "Experiments directory")
    configs.DEFINE_list_string("rnn_cell", 'lstm', "lstm or gru")
    configs.DEFINE_list_string("activation_fn", 'relu', "MLP activation function in tf.nn.*")
    configs.DEFINE_integer("num_inputs", -1, "")
    configs.DEFINE_integer("num_outputs", -1, "")
    configs.DEFINE_integer("target_idx", None, "")
    configs.DEFINE_list_integer("min_unrollings", 5, "Min number of unrolling steps")
    configs.DEFINE_list_integer("max_unrollings", 5, "Max number of unrolling steps")
    configs.DEFINE_list_integer("min_years", None, "Alt to min_unrollings")
    configs.DEFINE_list_integer("max_years", None, "Alt to max_unrollings")
    configs.DEFINE_integer("pls_years", None, "Alt to max_years. max_years = min_year+pls_years")
    configs.DEFINE_list_integer("stride", 12, "How many steps to skip per unrolling")

    configs.DEFINE_list_integer("batch_size", 256, "Size of each batch")
    configs.DEFINE_list_integer("num_layers", 2, "Numer of RNN layers")
    configs.DEFINE_integer("forecast_n", 12, "How many steps to forecast into the future")
    configs.DEFINE_list_integer("num_hidden", 64, "Number of hidden layer units")
    configs.DEFINE_list_float("init_scale", 1.0, "Initial scale for weights")
    configs.DEFINE_list_float("max_grad_norm", 50.0, "Gradient clipping")
    configs.DEFINE_integer("start_date", 197501, "First date to train on as YYYYMM")
    configs.DEFINE_integer("end_date", 199812, "Last date to train on as YYYYMM")
    configs.DEFINE_integer("split_date", None, "Date to split train/test on.")
    configs.DEFINE_boolean("train", True, "Train model otherwise inference only")
    configs.DEFINE_list_float("dropout", 0.0, "Dropout rate for hidden layers")
    configs.DEFINE_list_float("recurrent_dropout", 0.3, "Dropout rate for recurrent connections")
    configs.DEFINE_boolean("log_squasher", True, "Squash large normalized inputs with natural log function")
    configs.DEFINE_list_string("data_scaler", 'RobustScaler', 'sklearn scaling algorithm or None if no scaling')
    configs.DEFINE_list_string("optimizer", 'Adadelta', 'Any tensorflow optimizer in tf.train')
    configs.DEFINE_list_float("learning_rate", 0.6, "The initial starting learning rate")
    configs.DEFINE_list_float("lr_decay", 0.96, "Learning rate decay")
    configs.DEFINE_float("validation_size", 0.3, "Size of validation set as %, ie. 0.3 = 30% of data")
    configs.DEFINE_list_float("target_lambda", 0.5, "How much to weight last step vs. all steps in loss")
    configs.DEFINE_list_float("rnn_lambda", 0.7, "How much to weight last step vs. all steps in loss")
    configs.DEFINE_integer("max_epoch", 35, "Stop after max_epochs")
    configs.DEFINE_integer("early_stop", 15, "Early stop parameter")
    configs.DEFINE_integer("seed", 521, "Seed for deterministic training")
    configs.DEFINE_boolean("UQ", False, "Uncertainty Quantification Mode")
    configs.DEFINE_list_float("l2_alpha", 0.0, "L2 regularization for weight parameters.")
    configs.DEFINE_float("recurrent_l2_alpha", 0.0, "L2 regularization for recurrent weight parameters.")
    configs.DEFINE_list_boolean("huber_loss", False, "Use huber loss instead of mse")
    configs.DEFINE_list_float("huber_delta", 1.0, "delta for huber loss")
    configs.DEFINE_integer("forecast_steps", 1, "How many future predictions need to me made")
    configs.DEFINE_string('forecast_steps_weights', '1.0', 'weights for the forecast steps')
    configs.DEFINE_integer("logging_interval", 100, "Number of batches for logging interval during training")
    configs.DEFINE_boolean("write_inp_to_out_file", True, "Write input sequence to the output files")
    configs.DEFINE_string("training_type", 'fixed_dates', 'Choose between "fixed_dates" and "iterative" training')
    configs.DEFINE_integer("member_id", 1, "Id of member in a population")
    configs.DEFINE_boolean('load_saved_weights', False, 'Load weights saved in the checkpoint directory')
    configs.DEFINE_integer("epoch_logging_interval", 1, "Number of batches for logging interval during training")
    configs.DEFINE_string('preds_fname', 'preds.dat', 'Name of the prediction file')
    configs.DEFINE_integer("num_procs", 1, "Total number of training/prediction processes")

    # HPO related params
    configs.DEFINE_integer("NPE", 1, "Number of Parallel Executions")
    configs.DEFINE_string("search_algorithm", "genetic",
                          "Algorithm for hyper-param optimization. Select from 'genetic', 'grid_search', 'doe' ")
    configs.DEFINE_integer("generations", 5, "Number of generations for genetic algorithm")
    configs.DEFINE_integer("pop_size", 16, "Population size for genetic algorithm")
    configs.DEFINE_integer("num_gpu", 1, "Number of GPU on the machine, Use 0 if there are None")
    configs.DEFINE_float("mutate_rate", 0.2, "Mutation rate for genetic algorithm")
    configs.DEFINE_string("objective", 'mse', "Select between mse or uq_loss")
    configs.DEFINE_string("init_pop", None, "Initial population to begin hyper param search")
    configs.DEFINE_boolean("save_latest_pop", False, "Save the latest population")
    configs.DEFINE_string('doe_file', None, 'Design of experiments csv file')
    configs.DEFINE_integer("decay_steps", 100000, "Number of training steps between decay steps")
    configs.DEFINE_string("initializer", 'GlorotUniform', 'variable initializers available in Keras')
    configs.DEFINE_boolean("use_custom_init", True, 'Use RandomUniform initializer with init_scale values')
    configs.DEFINE_boolean("aux_masking", False, 'Mask aux features of all time steps except the last one with 0')
    configs.DEFINE_integer("max_norm", None, "Max Norm for kernel constraint")
    configs.DEFINE_float("sgd_momentum", 0.0, "momentum for SGD optimizer")
    configs.DEFINE_float("end_learning_rate", 0.01, "end lr for polynomial decay")
    configs.DEFINE_float('decay_power', 0.5, 'power to decay the learning rate with for polynomial decay')
    configs.DEFINE_string('piecewise_lr_boundaries', None, 'boundaries for piecewise constant lr')
    configs.DEFINE_string('piecewise_lr_values', None, 'values for piecewise constant lr')
    configs.DEFINE_string('lr_schedule', 'ExponentialDecay', 'Learning rate scheduler')

    c = configs.ConfigValues()

    c.data_dir = os.path.join(_data_dir_path, c.data_dir)
    c.forecast_steps_weights = [float(x) for x in c.forecast_steps_weights.split(',')]

    return c


def main():
    config = get_configs()
    # cc = config.__dict__['__configs']
    # for k, v in cc.items():
    #     print(k, v)

    hpo = HPOptimization(config)
    hpo()

    # pool = mp.Pool(config.NPE)
    # results = pool.map(execute_case, configs_list)


if __name__ == "__main__":
    main()
