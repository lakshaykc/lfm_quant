"""Implementation of the configs interface."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


class _LoadFromFile(argparse.Action):
    """Helper that supports the reading of config from a file"""

    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            parser.parse_known_args(f.read().split(), namespace)


_global_parser = argparse.ArgumentParser()

_global_parser.add_argument('--config', type=open,
                            action=_LoadFromFile,
                            help="File containing configuration")


class ConfigValues(object):
    """
    Command line argument helper class.
    """

    def __init__(self):
        """Global container and accessor for configs and their values."""
        self.__dict__['__configs'] = {}
        self.__dict__['__parsed'] = False

    def _parse_configs(self):
        result, _ = _global_parser.parse_known_args()
        if '__configs' not in self.__dict__:
            self.__dict__['__configs'] = {}
        if '__parsed' not in self.__dict__:
            self.__dict__['__parsed'] = False
        for config_name, val in vars(result).items():
            self.__dict__['__configs'][config_name] = val
        self.__dict__['__parsed'] = True

    def __getattr__(self, name):
        """Retrieves the 'value' attribute of the config --name."""
        if ('__parsed' not in self.__dict__) or (not self.__dict__['__parsed']):
            self._parse_configs()
        if name not in self.__dict__['__configs']:
            raise AttributeError(name)
        return self.__dict__['__configs'][name]

    def __setattr__(self, name, value):
        """Sets the 'value' attribute of the config --name."""
        if ('__parsed' not in self.__dict__) or (not self.__dict__['__parsed']):
            self._parse_configs()
        self.__dict__['__configs'][name] = value


def _define_helper(config_name, default_value, docstring, configtype):
    """Registers 'config_name' with 'default_value' and 'docstring'."""
    _global_parser.add_argument("--" + config_name,
                                default=default_value,
                                help=docstring,
                                type=configtype)


def DEFINE_string(config_name, default_value, docstring):
    """Defines a config of type 'string'.
    Args:
      config_name: The name of the config as a string.
      default_value: The default value the config should take as a string.
      docstring: A helpful message explaining the use of the config.
    """
    _define_helper(config_name, default_value, docstring, str)


def DEFINE_integer(config_name, default_value, docstring):
    """Defines a config of type 'int'.
    Args:
      config_name: The name of the config as a string.
      default_value: The default value the config should take as an int.
      docstring: A helpful message explaining the use of the config.
    """
    _define_helper(config_name, default_value, docstring, int)


def DEFINE_boolean(config_name, default_value, docstring):
    """Defines a config of type 'boolean'.
    Args:
      config_name: The name of the config as a string.
      default_value: The default value the config should take as a boolean.
      docstring: A helpful message explaining the use of the config.
    """

    # Register a custom function for 'bool' so --config=True works.
    def str2bool(v):
        return v.lower() in ('true', 't', '1')

    _global_parser.add_argument('--' + config_name,
                                nargs='?',
                                const=True,
                                help=docstring,
                                default=default_value,
                                type=str2bool)
    _global_parser.add_argument('--no' + config_name,
                                action='store_false',
                                dest=config_name)


# The internal google library defines the following alias, so we match
# the API for consistency.
DEFINE_bool = DEFINE_boolean  # pylint: disable=invalid-name


def DEFINE_float(config_name, default_value, docstring):
    """Defines a config of type 'float'.
    Args:
      config_name: The name of the config as a string.
      default_value: The default value the config should take as a float.
      docstring: A helpful message explaining the use of the config.
    """
    _define_helper(config_name, default_value, docstring, float)


def DEFINE_list_integer(config_name, default_value, docstring):
    """
    Defines a config of a list with `int` data type
    :param config_name: The name of the config as a string
    :param default_value: The default value the config should take as an int
    :param docstring: A helpful message explaining the use of the config.
    :return:
    """

    _global_parser.add_argument('--' + config_name,
                                nargs='*',
                                help=docstring,
                                default=[int(default_value)] if default_value is not None else [None],
                                type=int)


def DEFINE_list_float(config_name, default_value, docstring):
    """
    Defines a config of a list with `float` data type
    :param config_name: The name of the config as a string
    :param default_value: The default value the config should take as an float
    :param docstring: A helpful message explaining the use of the config.
    :return:
    """

    _global_parser.add_argument('--' + config_name,
                                nargs='*',
                                help=docstring,
                                default=[float(default_value)] if default_value is not None else [None],
                                type=float)


def DEFINE_list_string(config_name, default_value, docstring):
    """
    Defines a config of a list with `str` data type
    :param config_name: The name of the config as a string
    :param default_value: The default value the config should take as an str
    :param docstring: A helpful message explaining the use of the config.
    :return:
    """

    _global_parser.add_argument('--' + config_name,
                                nargs='*',
                                help=docstring,
                                default=[default_value],
                                type=str)


def DEFINE_list_boolean(config_name, default_value, docstring):
    """
    Defines a config of a list with `boolean` data type
    :param config_name: The name of the config as a string
    :param default_value: The default value the config should take as a boolean
    :param docstring: A helpful message explaining the use of the config.
    :return:
    """
    def convert_to_list(s):
        return s.lower() in ('true', 't', '1')

    _global_parser.add_argument('--' + config_name,
                                nargs='*',
                                help=docstring,
                                default=[default_value],
                                type=convert_to_list)
