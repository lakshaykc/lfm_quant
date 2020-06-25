from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import configs


def get_configs():
    """
    Defines all configuration params passable to command line.
    """
    configs.DEFINE_string("name", 'test', "A name for the config.")
    configs.DEFINE_string("datafile", None, "a datafile name.")
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
    configs.DEFINE_string("dont_scale_fields", None, "Names of fields to not scale")
    configs.DEFINE_string("data_dir", 'datasets', "The data directory")
    configs.DEFINE_string("model_dir", 'test-model', "Model directory")
    configs.DEFINE_string("experiments_dir", './', "Experiments directory")
    configs.DEFINE_string("rnn_cell", 'lstm', "lstm or gru")
    configs.DEFINE_string("activation_fn", 'relu', "MLP activation function in tf.nn.*")
    configs.DEFINE_integer("num_inputs", -1, "")
    configs.DEFINE_integer("num_outputs", -1, "")
    configs.DEFINE_integer("target_idx", None, "")
    configs.DEFINE_integer("min_unrollings", 5, "Min number of unrolling steps")
    configs.DEFINE_integer("max_unrollings", 5, "Max number of unrolling steps")
    configs.DEFINE_integer("min_years", None, "Alt to min_unrollings")
    configs.DEFINE_integer("max_years", None, "Alt to max_unrollings")
    configs.DEFINE_integer("pls_years", None, "Alt to max_years. max_years = min_year+pls_years")
    configs.DEFINE_integer("stride", 12, "How many steps to skip per unrolling")
    configs.DEFINE_integer("batch_size", 256, "Size of each batch")
    configs.DEFINE_integer("num_layers", 2, "Numer of RNN layers")
    configs.DEFINE_integer("forecast_n", 12, "How many steps to forecast into the future")
    configs.DEFINE_integer("num_hidden", 64, "Number of hidden layer units")
    configs.DEFINE_float("init_scale", 1.0, "Initial scale for weights")
    configs.DEFINE_float("max_grad_norm", 50.0, "Gradient clipping")
    configs.DEFINE_integer("start_date", 197501, "First date to train on as YYYYMM")
    configs.DEFINE_integer("end_date", 199912, "Last date to train on as YYYYMM")
    configs.DEFINE_integer("split_date", None, "Date to split train/test on.")
    configs.DEFINE_boolean("train", True, "Train model otherwise inference only")
    configs.DEFINE_float("dropout", 0.0, "Dropout rate for hidden layers")
    configs.DEFINE_float("recurrent_dropout", 0.0, "Dropout rate for recurrent connections")
    configs.DEFINE_boolean("log_squasher", True, "Squash large normalized inputs with natural log function")
    configs.DEFINE_string("data_scaler", 'RobustScaler', 'sklearn scaling algorithm or None if no scaling')
    configs.DEFINE_string("optimizer", 'Adadelta', 'Any tensorflow optimizer in tf.train')
    configs.DEFINE_float("learning_rate", 0.6, "The initial starting learning rate")
    configs.DEFINE_float("lr_decay", 1.0, "Learning rate decay for exponential decay")
    configs.DEFINE_float("validation_size", 0.3, "Size of validation set as %, ie. 0.3 = 30% of data")
    configs.DEFINE_float("target_lambda", 0.5, "How much to weight last step vs. all steps in loss")
    configs.DEFINE_float("rnn_lambda", 0.7, "How much to weight last step vs. all steps in loss")
    configs.DEFINE_integer("max_epoch", 1, "Stop after max_epochs")
    configs.DEFINE_integer("early_stop", 1, "Early stop parameter")
    configs.DEFINE_integer("seed", 521, "Seed for deterministic training")
    configs.DEFINE_boolean("UQ", False, "Uncertainty Quantification Mode")
    configs.DEFINE_float("l2_alpha", 0.0, "L2 regularization for weight parameters.")
    configs.DEFINE_float("recurrent_l2_alpha", 0.0, "L2 regularization for recurrent weight parameters.")
    configs.DEFINE_boolean("huber_loss", False, "Use huber loss instead of mse")
    configs.DEFINE_float("huber_delta", 1.0, "delta for huber loss")
    configs.DEFINE_integer("forecast_steps", 1, "How many future predictions need to me made")
    configs.DEFINE_string('forecast_steps_weights', '1.0', 'weights for the forecast steps')
    configs.DEFINE_integer("logging_interval", 100, "Number of batches for logging interval during training")
    configs.DEFINE_boolean("write_inp_to_out_file", True, "Write input sequence to the output files")
    configs.DEFINE_string("training_type", 'fixed_dates', 'Choose between "fixed_dates" and "iterative" training')
    configs.DEFINE_integer("NPE", 1, "Number of Parallel Executions")
    configs.DEFINE_integer("num_procs", 1, "Total number of training/prediction processes")
    configs.DEFINE_integer("num_gpu", 1, "NUmber of GPUs")
    configs.DEFINE_boolean('load_saved_weights', False, 'Load weights saved in the checkpoint directory')
    configs.DEFINE_integer("epoch_logging_interval", 1, "Number of batches for logging interval during training")
    configs.DEFINE_integer("decay_steps", 1500, "Number of training steps between decay steps")
    configs.DEFINE_string("initializer", 'GlorotUniform', 'variable initializers available in Keras')
    configs.DEFINE_boolean("use_custom_init", True, 'Use RandomUniform initializer with init_scale values')
    configs.DEFINE_boolean("aux_masking", False, 'Mask aux features of all time steps except the last one with 0')
    configs.DEFINE_integer("max_norm", 3, "Max Norm for kernel constraint")
    configs.DEFINE_float("sgd_momentum", 0.0, "momentum for SGD optimizer")
    configs.DEFINE_float("end_learning_rate", 0.01, "end lr for polynomial decay")
    configs.DEFINE_float('decay_power', 0.5, 'power to decay the learning rate with for polynomial decay')
    configs.DEFINE_string('piecewise_lr_boundaries', '4000-5500-5500', 'boundaries for piecewise constant lr')
    configs.DEFINE_string('piecewise_lr_values', '0.5-0.1-0.05-0.1', 'values for piecewise constant lr')
    configs.DEFINE_string('lr_schedule', 'ExponentialDecay', 'Learning rate scheduler')
    configs.DEFINE_string('preds_fname', 'preds.dat', 'Name of the prediction file')
    configs.DEFINE_integer("member_id", 1, "Id of member in a population")
    configs.DEFINE_boolean("cdrs_inference", False, 'If the execution is for inference on CDRS data')
    configs.DEFINE_boolean("use_external_cdrs_data", False,
                           'True if CDRS data is provided externally. Otherwise it is generated internally ')
    configs.DEFINE_string('cdrs_src_fname', 'cdrs-src.dat', 'Filename of the CDRS source file')
    configs.DEFINE_string('cdrs_ml_fname', 'cdrs-ml-data.dat', 'Filename of the CDRS ML data file')
    configs.DEFINE_string('model_ranking_fname', './model-ranking.dat', 'Model Ranking File Name')
    configs.DEFINE_string('model_ranking_factor', 'pred_var_entval', 'Model ranking factor')

    c = configs.ConfigValues()

    if c.min_unrollings is None:
        c.min_unrollings = c.num_unrollings

    if c.max_unrollings is None:
        c.max_unrollings = c.num_unrollings

    if c.min_years is not None:
        c.min_unrollings = c.min_years * (12 // c.stride)
        if c.max_years is not None:
            c.max_unrollings = (c.max_years) * (12 // c.stride)
        elif c.pls_years is None:
            c.max_unrollings = c.min_unrollings
        else:
            c.max_unrollings = (c.min_years + c.pls_years) * (12 // c.stride)

    c.forecast_steps_weights = [float(x) for x in c.forecast_steps_weights.split(',')]

    return c
