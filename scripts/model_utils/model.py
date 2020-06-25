from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# All models should be imported here
from models.point_estimate.rnn_point_estimate import RNNPointEstimate
from models.point_estimate.mlp_point_estimate import MLPPointEstimate
from models.point_estimate.mlp_linear_point_estimate import MLPLinearPointEstimate
from models.point_estimate.naive_point_estimate import NaivePointEstimate
from models.uq_range_estimate.rnn_uq_range_estimate import RNNUqRangeEstimate
from models.uq_range_estimate.mlp_uq_range_estimate import MLPUqRangeEstimate


class Model(object):

    def __init__(self, config, dataset):
        self._config = config
        self._dataset = dataset

    def get_model(self):
        """
        Creates model based on nn_type specified in config
        :return: model instance
        """
        all_objects = globals()
        if self._config.nn_type in all_objects:
            model_constructor = all_objects[self._config.nn_type]
        else:
            raise RuntimeError("Unknown nn_type = %s" % self._config.nn_type)

        m = model_constructor(self._config, self._dataset)
        model = m.model
        print(model.summary())

        if self._config.UQ:
            assert 'uq' in self._config.nn_type.lower(), "UQ should be True only for UQ Models"
        else:
            assert 'point' in self._config.nn_type.lower(), "UQ should be False for Point Estimate Models"
        return model
