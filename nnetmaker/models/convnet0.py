# -*- coding: utf-8 -*-
"""
Defines a convolutional neural network with residual connections.

Based on the architecture described in:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep residual learning
for image recognition". https://arxiv.org/abs/1512.03385

With batch normalization as described in:
Sergey Ioffe, Christian Szegedy. "Batch normalization: Accelerating
deep network training by reducing internal covariate shift".
https://arxiv.org/abs/1502.03167

And parametric ReLU activations as described in:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Delving Deep into
Rectifiers: Surpassing Human-Level Performance on ImageNet Classification".
https://arxiv.org/abs/1502.01852
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np
import tensorflow as tf

from nnetmaker.model import *
from nnetmaker.util import *


class ConvNetClassifierModel0(BaseModel):



  def _process_args(self, model_args_validator, **kwargs):
    self._learn_alpha = model_args_validator.get("learn_alpha", ATYPE_BOOL, True)
    self._alpha = model_args_validator.get("alpha", ATYPE_FLOAT, True)
    self._dropout_rate = model_args_validator.get("dropout_rate", ATYPE_FLOAT, True)
    self._conv_layer_sizes = model_args_validator.get("conv_layer_sizes", ATYPE_INTS_LIST, True)
    self._conv_layer_dims = model_args_validator.get("conv_layer_dims", ATYPE_INTS_LIST, True)
    self._fc_layer_dims = model_args_validator.get("fc_layer_dims", ATYPE_INTS_LIST, True)
    self._num_input_channels = model_args_validator.get("num_input_channels", ATYPE_INT, True)
    self._input_size = model_args_validator.get("input_size", ATYPE_INT, True)
    self._output_size = model_args_validator.get("output_size", ATYPE_INT, True)
    self._add_biases = model_args_validator.get("add_biases", ATYPE_BOOL, True)





  def _get_input_var_names(self, **kwargs):
    return ["img"]





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


    

    # Build convolutional layers.
    prev_var = input_vars["img"]
    prev_dims = self._num_input_channels

    for i in range(len(self._conv_layer_sizes)):
      size = self._conv_layer_sizes[i]
      cur_dims = self._conv_layer_dims[i]

      h_var = self._add_op_square_conv2d(prev_var, "conv%d" % i, weight_vars,
                                         weight_init_tups, self._add_biases,
                                         prev_dims, cur_dims, size)
      
      h_var = self._add_op_batch_norm(h_var, "norm%d" % i, 3, is_training)

      h_var = self._add_op_relu(h_var, "relu%d" % i, alpha=self._alpha,
                                is_variable=self._learn_alpha)

      # Add zero padded residual connection.
      if cur_dims > prev_dims:
        num_zeros = cur_dims - prev_dims
        paddings = np.zeros((4, 2), dtype=int)
        paddings[3, 1] = num_zeros
        h_var = h_var + tf.pad(prev_var, paddings)
      else:
        h_var = h_var + prev_var[:cur_dims]

      prev_dims = cur_dims
      prev_var = h_var



    # Build fully connected and output layers.
    h_var = prev_var
    h_size = self._input_size

    for i, cur_dims in enumerate(self._fc_layer_dims + [self._output_size]):

      if self._dropout_rate > 0:
        h_var = self._add_op_dropout(h_var, "dropout%d" % i, self._dropout_rate,
                                     is_training)


      h_var = self._add_op_square_conv2d(h_var, "fc%d" % i, weight_vars,
                                         weight_init_tups, self._add_biases,
                                         prev_dims, cur_dims, h_size, pad=False)
      h_size = 1
      prev_dims = cur_dims

      if i < len(self._fc_layer_dims):   # Hidden fully connected layer.
        h_var = self._add_op_batch_norm(h_var, "fc_norm%d" % i, 3, is_training)
        h_var = self._add_op_relu(h_var, "fc_relu%d" % i, alpha=self._alpha,
                                  is_variable=self._learn_alpha)

      else:   # Final output layer.
        h_var = tf.reduce_mean(h_var, axis=1)
        h_var = tf.reduce_mean(h_var, axis=1)
        h_var = tf.nn.softmax(h_var)


    out_vars = {}
    out_vars["predictions"] = h_var


    return out_vars, weight_vars, weight_init_tups



