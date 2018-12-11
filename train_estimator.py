# -*- coding: utf-8 -*-
"""
Trains a TensorFlow Estimator model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Filter everything but errors.

import logging
import numpy as np
import nnetmaker as nn
import tensorflow as tf
import traceback







class TrainCheckpointSaverListener(tf.train.CheckpointSaverListener):
  """CheckpointSaverListener that just logs an info message every save."""

  def __init__(self):
    self._logger = logging.getLogger()

  def after_save(self, session, global_step_value):
    self._logger.info("%d: Wrote checkpoint." % global_step_value)










def run_training(model, train_loader, val_loader, params, logger, np_rand_state):
  """Runs training and evaluation until stopped."""

  run_config = tf.estimator.RunConfig(model_dir=model.model_dir,
                                      tf_random_seed=np_rand_state.randint(12345, 2**32),
                                      save_checkpoints_secs=params["save_time"])

  estimator = tf.estimator.Estimator(model_fn=model.build_model_fn(),
                                     config=run_config, params=params)


  try:

    train_spec = tf.estimator.TrainSpec(input_fn=train_loader.build_input_fn())
    eval_spec = tf.estimator.EvalSpec(input_fn=val_loader.build_input_fn(),
                                      steps=None, throttle_secs=params["val_throttle_time"])

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


  except KeyboardInterrupt:
    logger.info("Stopped by user.")






def main(config_filename):

  try:

    # Parse and validate the specified configuration.
    tup = nn.load_config(config_filename)
    rand_seed = tup[0]
    epsilon = tup[1]
    model_type = tup[2]
    model_args = tup[3]
    input_args = tup[4]
    init_args = tup[5]
    train_args = tup[6]
    logger_args = tup[7]

    # Initialize the logger.
    nn.configure_logger(logger_args)
    logger = logging.getLogger()


    # Initialize random number generators.
    tf_rand_seed, np_rand_seed = nn.configure_seeds(2, rand_seed)
    tf.set_random_seed(tf_rand_seed)
    np_rand_state = np.random.RandomState(np_rand_seed)


    # Initialize the model and input loaders.
    model = nn.MODELS[model_type](model_args, epsilon=epsilon)
    train_loader = nn.configure_input(model.input_names, model.target_names,
                                      input_args, nn.INPUT_TRAIN)
    val_loader = nn.configure_input(model.input_names, model.target_names,
                                    input_args, nn.INPUT_VAL)
    params = nn.configure_estimator_params(init_args, train_args)
    assert train_loader.target_batch_size == val_loader.target_batch_size
    model._batch_size = train_loader.target_batch_size

    # Run training loop.
    run_training(model, train_loader, val_loader, params, logger, np_rand_state)



  except Exception as ex:
    logger = logging.getLogger()
    ex_str = traceback.format_exc()
    logger.error(ex_str)
    exit(1)





if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description=__doc__)

  parser.add_argument("config_filename", help="Filename of json model config file")
  
  args = parser.parse_args()
  main(os.path.realpath(args.config_filename))





