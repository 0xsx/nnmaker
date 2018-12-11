# -*- coding: utf-8 -*-
"""
Exports a TensorFlow Estimator model to a frozen model protobuf and json header.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Filter everything but errors.

import json
import logging
import numpy as np
import nnetmaker as nn
import tensorflow as tf
import traceback








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


    # Initialize the model and input loader.
    model = nn.MODELS[model_type](model_args, epsilon=epsilon)
    loader = nn.configure_input(model.input_names, model.target_names,
                                input_args, nn.INPUT_TEST)
    params = nn.configure_estimator_params(init_args, train_args)
    model._batch_size = loader.target_batch_size


    # Restore and freeze model.
    with open(os.path.join(model.model_dir, "checkpoint"), "r") as f_in:
      latest_checkpoint = f_in.readline().split(":")[1].strip()[1:-1]


    with tf.Session() as tf_session:
      input_fn = loader.build_input_fn()
      final_inputs, final_targets = tf_session.run(input_fn())


      placeholders = {}
      for name in model.input_names:
        placeholders[name] = tf.placeholder(final_inputs[name].dtype,
                                            shape=final_inputs[name].shape,
                                            # shape=[1, None, None, 1],
                                            name=name)

      tup = model.build_predict_graph(placeholders)
      in_tensor_names = tup[0]
      out_tensor_names = tup[1]
      out_op_names = tup[2]

      saver = tf.train.Saver()
      saver.restore(tf_session, os.path.join(model.model_dir, latest_checkpoint))

      graph = tf_session.graph
      graph.finalize()
      graph_def = graph.as_graph_def()


      # graph_def = tf.graph_util.remove_training_nodes(graph_def)
      graph_def = tf.graph_util.convert_variables_to_constants(tf_session,
                                                               graph_def,
                                                               out_op_names) 
    

      for node in graph_def.node:
        node.device = ""


      frozen_json = {}
      frozen_json["input_tensors"] = in_tensor_names
      frozen_json["output_tensors"] = out_tensor_names

      with tf.gfile.FastGFile(os.path.join(model.model_dir, "frozen.pb"), "wb") as f_out:
        f_out.write(graph_def.SerializeToString())

      with open(os.path.join(model.model_dir, "frozen.json"), "w") as f_out:
        json.dump(frozen_json, f_out, indent=1)
        f_out.write("\n")




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





