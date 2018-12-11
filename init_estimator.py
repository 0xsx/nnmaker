# -*- coding: utf-8 -*-
"""
Initializes and saves new model weights to be trained by the training program.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Filter everything but errors.

import logging
import numpy as np
import nnetmaker as nn
import shutil
import tensorflow as tf
import traceback





def init_ortho_weights(shape, rand_state, alpha=1.0, dtype="float64", eps=float(1e-8)):
  """Initializes orthonormal weights for a layer as described in Saxe et al. 2014.
  If the first dimension is smaller than or equal to the second dimension of the
  shape when flattened to 2 dimensions, each entry along the first dimension will be
  orthogonal to the others. Otherwise, columns in the array flattened to 2
  dimensions will be orthogonal. The `alpha` variable is used to scale weights by
  `s = sqrt(2/(1+alpha**2))` and should correspond to the alpha value used for
  leaky ReLU, i.e. 0 for standard ReLU units. See Saxe et al. 2014. "Exact
  solutions to the nonlinear dynamics of learning in deep linear neural
  networks." for details: http://arxiv.org/abs/1312.6120"""

  flat_shape = (shape[0], np.prod(shape[1:]))
  w = rand_state.randn(*flat_shape)
  w -= np.mean(w)
  w /= (np.std(w) + eps)

  u, _, v = np.linalg.svd(w, full_matrices=False)
  w = u if u.shape == flat_shape else v
  w = w.reshape(shape)

  s = np.sqrt(2. / (1. + alpha**2))

  return (s * w).astype(dtype)






def run_init(model, weight_init_tups, weights_rand_state, params,
             logger, epsilon, tf_session):
  """Initializes orthonormal weights and performs layer-sequential unit-variance
  orthogonal initialization on them. See Mishkin and Matas. 2016. "All you
  need is a good init." for details about LSUV: https://arxiv.org/abs/1511.06422"""



  # Count total weights and assign Gaussian random weights.
  layer_weight_arrays = []
  num_weights = 0
  num_lsuv_weights = 0
  for w_shape, w_assign_fn, _, is_lsuv, _, dtype in weight_init_tups:
    w_arr = 0.1 * weights_rand_state.randn(*w_shape)
    w_arr = w_arr.astype(dtype)
    w_assign_fn(w_arr, tf_session)
    layer_weight_arrays.append(w_arr)
    num_weights += len(w_arr)
    if is_lsuv:
      num_lsuv_weights += len(w_arr)



  # Initialize orthonormal weight matrices.
  if params["use_ortho_weights"]:
    layer_weight_arrays = []
    cur_num_weights = 0

    for w_shape, w_assign_fn, _, _, out_name, dtype in weight_init_tups:
      w_arr = init_ortho_weights(w_shape, weights_rand_state, alpha=params["init_alpha"])
      w_arr = w_arr.astype(dtype)
      w_assign_fn(w_arr, tf_session)
      layer_weight_arrays.append(w_arr)
      cur_num_weights += len(w_arr)


      if w_shape[0] > np.prod(w_shape[1:]):
        logger.warn("Weights for output \"%s\" not completely orthogonal." % out_name)


      logger.info("%d / %d weights orthonormal initialized..." % (cur_num_weights, num_weights))






  # Initialize LSUV weights.
  if params["max_lsuv_iters"] is not None:
    logger.info("Performing layer-sequential unit-variance initialization...")

    cur_num_weights = 0

    for w_arr, (_, w_assign_fn, out_var, allow_lsuv, out_name, dtype) in zip(layer_weight_arrays, weight_init_tups):
      if not allow_lsuv:
        continue

      converged = False
      best_var = float("inf")
      for _ in range(params["max_lsuv_iters"]):
        cur_var = np.var(tf_session.run(out_var))

        if np.isnan(cur_var):
          logger.error("NaN layer variance encountered. Exiting.")
          exit(1)

        if np.fabs(cur_var - 1.0) < params["lsuv_tolerance"]:
          converged = True
          break
        best_var = cur_var

        w_arr /= (np.sqrt(cur_var + epsilon) + epsilon)
        w_arr = w_arr.astype(dtype)
        w_assign_fn(w_arr, tf_session)


      if not converged:
        logger.warn("Layer variance for output \"%s\" did not converge. Reached %.4f." % (out_name, best_var))

      cur_num_weights += len(w_arr)
      logger.info("%d / %d weights LSUV initialized..." % (cur_num_weights, num_lsuv_weights))


  









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
    weights_rand_state = np.random.RandomState(np_rand_seed)


    # Initialize the model and input loader.
    model = nn.MODELS[model_type](model_args, epsilon=epsilon)
    loader = nn.configure_input(model.input_names, model.target_names,
                                input_args, nn.INPUT_TRAIN)
    params = nn.configure_estimator_params(init_args, train_args)
    model._batch_size = loader.target_batch_size


    # Check and delete directories if configured to do so.
    try:
      is_model_dir_empty = len(os.listdir(model.model_dir)) == 0
    except:
      is_model_dir_empty = True

    if not is_model_dir_empty:
      if not params["rm_dir_on_init"]:
        logger.error("Model dir not empty. Exiting.")
        exit(1)
      else:
        logger.warn("Model dir not empty, deleting...")
        shutil.rmtree(model.model_dir)

    try:
      os.makedirs(model.model_dir)
    except OSError: pass




    

    with tf.Session() as tf_session:

      # Construct initialization graph and initialize all variables with their
      # default initializers.
      input_fn = loader.build_input_fn()
      init_inputs, init_targets = tf_session.run(input_fn())
      weight_init_tups = model.build_weight_init_tups(init_inputs, init_targets, params)

      tf_session.run(tf.local_variables_initializer())
      tf_session.run(tf.global_variables_initializer())
      

      # Initialize weights according to configured parameters and save weights.
      run_init(model, weight_init_tups, weights_rand_state, params,
               logger, epsilon, tf_session)
      saver = tf.train.Saver()
      saver.save(tf_session, os.path.join(model.model_dir, "model"))
      logger.info("Saved model weights.")


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




