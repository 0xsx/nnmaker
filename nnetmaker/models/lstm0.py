# -*- coding: utf-8 -*-
"""
Defines a deep bi-directional LSTM neural network with peephole
connections for classification.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np
import tensorflow as tf

from nnetmaker.model import *
from nnetmaker.util import *





class LSTMClassifierModel0(BaseModel):



  def _process_args(self, model_args_validator, **kwargs):
    self._learn_alpha = model_args_validator.get("learn_alpha", ATYPE_BOOL, True)
    self._alpha = model_args_validator.get("alpha", ATYPE_FLOAT, True)
    self._dropout_rate = model_args_validator.get("dropout_rate", ATYPE_FLOAT, True)
    self._output_size = model_args_validator.get("output_size", ATYPE_INT, True)
    self._fc_layer_dims = model_args_validator.get("fc_layer_dims", ATYPE_INTS_LIST, True)
    self._hidden_layer_dims = model_args_validator.get("hidden_layer_dims", ATYPE_INTS_LIST, True)
    self._feat_dims = model_args_validator.get("feat_dims", ATYPE_INT, True)
    self._timesteps = model_args_validator.get("timesteps", ATYPE_INT, True)



  def _get_input_var_names(self, **kwargs):
    return ["feats"]





  def _get_target_var_names(self, **kwargs):
    return ["predictions"]





  def _build_cost_targets(self, in_vars, target_vars, out_vars, **kwargs):

    cost_targets = []
    cost_targets.append((target_vars["predictions"], out_vars["predictions"], None))

    return cost_targets






  def _build_metrics(self, in_vars, target_vars, out_vars, **kwargs):
    metrics = {}

    targets = tf.argmax(target_vars["predictions"], axis=1)
    predicted = tf.argmax(out_vars["predictions"], axis=1)

    metrics["accuracy"] = tf.metrics.accuracy(targets, predicted)
    
    return metrics






  def _build_prediction_network(self, input_vars, is_training, **kwargs):

    weight_vars = []
    weight_init_tups = []


    # Transpose input features sequence so timesteps are along the first dimension.
    transposed_feats_var = tf.transpose(input_vars["feats"], (1, 0, 2))
    prev_dims = self._feat_dims
    feats_seq_var = transposed_feats_var



    # Bi-directional LSTM layers.
    for i, cur_dims in enumerate(self._hidden_layer_dims):

      fw_cell = tf.contrib.rnn.LSTMBlockCell(cur_dims, use_peephole=True, name="fw_%d" % i)
      bw_cell = tf.contrib.rnn.LSTMBlockCell(cur_dims, use_peephole=True, name="bw_%d" % i)


      outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, feats_seq_var,
                      time_major=True, dtype=FLOAT_DTYPE, scope="lstm_%d" % i, 
                      sequence_length=np.array([self._timesteps] * self._batch_size))

      h_seq_var = tf.concat(outputs, 2)
      h_seq_var = tf.reshape(h_seq_var, (self._timesteps, self._batch_size, 2*cur_dims))
      if i < len(self._hidden_layer_dims) - 1:
        # Inner hidden layers get batch normalized and nonlinearized after concat.
        # h_seq_var = self._add_op_batch_norm(h_seq_var, "norm_0_%d" % i, 0, is_training)

        h_seq_var = self._add_op_relu(h_seq_var, "relu_%d" % i, alpha=self._alpha,
                                      is_variable=self._learn_alpha)

        # Inner layers also get zero-padded residual connections from input
        # sequence to output sequence.
        if 2*cur_dims > prev_dims:
          num_zeros = 2*cur_dims - prev_dims
          paddings = np.zeros((3, 2), dtype=int)
          paddings[2, 1] = num_zeros
          h_seq_var = h_seq_var + tf.pad(feats_seq_var, paddings)
        else:
          h_seq_var = h_seq_var + feats_seq_var[:, :, :2*cur_dims]

        # h_seq_var = self._add_op_batch_norm(h_seq_var, "norm_1_%d" % i, 0, is_training)


      prev_dims = 2*cur_dims
      feats_seq_var = h_seq_var







    # Mean pool, normalize, and nonlinearize final LSTM layer output.
    
    h_var = tf.reduce_mean(feats_seq_var, axis=0)
    h_var = tf.reshape(h_var, (self._batch_size, 1, 1, prev_dims))
    # h_var = self._add_op_batch_norm(h_var, "norm_mean", 3, is_training)
    h_var = self._add_op_relu(h_var, "relu_mean", alpha=self._alpha,
                              is_variable=self._learn_alpha)
    


    # Fully connected and output layers.
    for i, cur_dims in enumerate(self._fc_layer_dims + [self._output_size]):

      if self._dropout_rate > 0:
        h_var = self._add_op_dropout(h_var, "dropout%d" % i, self._dropout_rate,
                                     is_training)


      h_var = self._add_op_square_conv2d(h_var, "fc%d" % i, weight_vars,
                                         weight_init_tups, True,
                                         prev_dims, cur_dims, 1, pad=False)
      prev_dims = cur_dims

      if i < len(self._fc_layer_dims):   # Hidden fully connected layer.
        # h_var = self._add_op_batch_norm(h_var, "fc_norm%d" % i, 3, is_training)
        h_var = self._add_op_relu(h_var, "fc_relu%d" % i, alpha=self._alpha,
                                  is_variable=self._learn_alpha)

      else:   # Final output layer.
        h_var = tf.reduce_mean(h_var, axis=1)
        h_var = tf.reduce_mean(h_var, axis=1)
        h_var = tf.nn.softmax(h_var)


    out_vars = {}
    out_vars["predictions"] = h_var


    return out_vars, weight_vars, weight_init_tups



