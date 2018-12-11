# -*- coding: utf-8 -*-
"""
Defines methods for configuration.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


__all__ = ["INPUT_TRAIN", "INPUT_VAL", "INPUT_TEST", "load_config",
           "configure_logger", "configure_input",
           "configure_estimator_params", "configure_seeds"]


import json
import logging
import logging.handlers
import numpy as np
import os
import sys


from nnetmaker.batch import *
from nnetmaker.util import *


# Loader types dictionary.
_LOADERS = {"independent": IndependentBatchLoader,
            "continuous_sequence": ContinuousSequenceBatchLoader,
            "discrete_sequence": DiscreteSequenceBatchLoader}



# Input modes.
INPUT_TRAIN = "train"
INPUT_VAL = "val"
INPUT_TEST = "test"




def _read_json_file(filename):
  """Reads a json object from the specified file."""

  # Strip comments while keeping line numbers.
  s = ""
  with open(filename, "r") as f_in:
    for line in f_in:
      comment_pos = line.find("//")
      s += line[:comment_pos] + "\n"

  return json.loads(s)
  




def load_config(filename):
  """Reads and validates the configuration file."""


  filename = os.path.realpath(filename)
  sys.path.append(os.path.dirname(filename))

  config_obj = _read_json_file(filename)
  config_val = ArgumentsValidator(config_obj, "Configuration file")

  
  with config_val:
    rand_seed = config_val.get("rand_seed", [ATYPE_NONE, ATYPE_INT], True)
    epsilon = config_val.get("epsilon", ATYPE_FLOAT, True)

    model_type = config_val.get("model_type", ATYPE_STRING, True)
    model_args = config_val.get("model_args", [ATYPE_STRING, ATYPE_DICT], True)
    if not isinstance(model_args, dict):
      model_args = _read_json_file(model_args)

    input_args = config_val.get("input_args", [ATYPE_STRING, ATYPE_DICT], True)
    if not isinstance(input_args, dict):
      input_args = _read_json_file(input_args)

    init_args = config_val.get("init_args", [ATYPE_STRING, ATYPE_DICT], True)
    if not isinstance(init_args, dict):
      init_args = _read_json_file(init_args)

    train_args = config_val.get("training_args", [ATYPE_STRING, ATYPE_DICT], True)
    if not isinstance(train_args, dict):
      train_args = _read_json_file(train_args)

    logger_args = config_val.get("logger_args", [ATYPE_STRING, ATYPE_DICT], False, default={})
    if not isinstance(logger_args, dict):
      logger_args = _read_json_file(logger_args)


  return (rand_seed, epsilon, model_type, model_args, input_args,
          init_args, train_args, logger_args)






def configure_logger(logger_args):
  """Configures the global Python logger object."""


  logger_val = ArgumentsValidator(logger_args, "Logger arguments")

  with logger_val:
    show_debug_logs = logger_val.get("show_debug_logs", ATYPE_BOOL, False, default=False)
    show_date = logger_val.get("show_date", ATYPE_BOOL, False, default=True)
    syslog_path = logger_val.get("syslog_path", [ATYPE_NONE, ATYPE_STRING], False)

  logger = logging.getLogger()

  if show_debug_logs:
    logger.setLevel(logging.DEBUG)
  else:
    logger.setLevel(logging.INFO)

  if syslog_path is not None:
    handler = logging.handlers.SysLogHandler(address=syslog_path)
  else:
    handler = logging.StreamHandler(stream=sys.stdout)


  if show_date:
    date_str = "%Y-%m-%d %H:%M:%S"
  else:
    date_str = "%H:%M:%S"


  formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", date_str)
  handler.setFormatter(formatter)

  logger.addHandler(handler)






def configure_input(model_input_names, model_output_names, input_args, input_mode):
  """Validates the input arguments and constructs the specified loader to return."""


  input_val = ArgumentsValidator(input_args, "Input arguments")

  with input_val:
    train_loader_type = input_val.get("train_loader_type", ATYPE_STRING, True)
    train_loader_args = input_val.get("train_loader_args", [ATYPE_STRING, ATYPE_DICT], True)
  
    val_loader_type = input_val.get("val_loader_type", ATYPE_STRING, True)
    val_loader_args = input_val.get("val_loader_args", [ATYPE_STRING, ATYPE_DICT], True)
  
    test_loader_type = input_val.get("test_loader_type", ATYPE_STRING, True)
    test_loader_args = input_val.get("test_loader_args", [ATYPE_STRING, ATYPE_DICT], True)
  

  if input_mode == INPUT_TRAIN:
    if not isinstance(train_loader_args, dict):
      train_loader_args = _read_json_file(train_loader_args)
    loader_type = train_loader_type
    loader_args = train_loader_args

  elif input_mode == INPUT_VAL:
    if not isinstance(val_loader_args, dict):
      val_loader_args = _read_json_file(val_loader_args)
    loader_type = val_loader_type
    loader_args = val_loader_args

  elif input_mode == INPUT_TEST:
    if not isinstance(test_loader_args, dict):
      test_loader_args = _read_json_file(test_loader_args)
    loader_type = test_loader_type
    loader_args = test_loader_args


  try:
    return _LOADERS[loader_type](model_input_names, model_output_names, loader_args)
  except KeyError:
    raise ValueError("Unrecognized loader type: %s" % loader_type)





def configure_seeds(num_seeds, rand_seed):
  """Initializes seeds for random number generators from the specified random seed."""

  base_rng = np.random.RandomState(rand_seed)
  seeds = [base_rng.randint(12345, 2**32) for _ in range(num_seeds)]

  return seeds







def configure_estimator_params(init_args, train_args):
  """Validates the initialization and training arguments and constructs a
  `params` dictionary for creating a TensorFlow Estimator object."""

  params = {}

  init_val = ArgumentsValidator(init_args, "Initialization arguments")
  with init_val:
    params["rm_dir_on_init"] = init_val.get("rm_dir", ATYPE_BOOL, True)
    params["use_ortho_weights"] = init_val.get("use_ortho_weights", ATYPE_BOOL, True)
    params["max_lsuv_iters"] = init_val.get("max_lsuv_iters", [ATYPE_NONE, ATYPE_INT], True)
    params["lsuv_tolerance"] = init_val.get("lsuv_tolerance", ATYPE_FLOAT, True)
    params["init_alpha"] = init_val.get("init_alpha", ATYPE_FLOAT, True)


  train_val = ArgumentsValidator(train_args, "Training arguments")
  with train_val:
    params["save_time"] = train_val.get("save_time", ATYPE_FLOAT, True)
    params["val_throttle_time"] = train_val.get("val_throttle_time", ATYPE_FLOAT, True)
    params["learning_rate"] = train_val.get("learning_rate", ATYPE_FLOAT, True)
    params["sgd_momentum"] = train_val.get("sgd_momentum", [ATYPE_NONE, ATYPE_FLOAT], True)
    params["sgd_use_nesterov"] = train_val.get("sgd_use_nesterov", ATYPE_BOOL, True)
    params["use_rmsprop"] = train_val.get("use_rmsprop", ATYPE_BOOL, True)
    params["rmsprop_decay"] = train_val.get("rmsprop_decay", ATYPE_FLOAT, True)
    params["rmsprop_momentum"] = train_val.get("rmsprop_momentum", ATYPE_FLOAT, True)
    params["rmsprop_epsilon"] = train_val.get("rmsprop_epsilon", ATYPE_FLOAT, True)
    params["reg_weight_decay"] = train_val.get("reg_weight_decay", [ATYPE_NONE, ATYPE_FLOAT], True)
    params["cost_type"] = train_val.get("cost_type", ATYPE_STRING, True).lower()
    params["max_grad_norm"] = train_val.get("max_grad_norm", [ATYPE_NONE, ATYPE_FLOAT], True)
    params["parallel_grad_gate"] = train_val.get("parallel_grad_gate", ATYPE_BOOL, True)


  return params



