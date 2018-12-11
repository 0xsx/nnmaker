# -*- coding: utf-8 -*-
"""
Defines a deep pyramidal bottleneck residual convolutional neural network.

Based on the architecture described in:
Dongyoon Han, Jiwhan Kim, Junmo Kim. "Deep Pyramidal Residual Networks".
https://arxiv.org/abs/1610.02915

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




class PyramidNetRegressorModel0(BaseModel):



  def _process_args(self, model_args_validator, **kwargs):
    self._learn_alpha = model_args_validator.get("learn_alpha", ATYPE_BOOL, True)
    self._alpha = model_args_validator.get("alpha", ATYPE_FLOAT, True)
    self._dropout_rate = model_args_validator.get("dropout_rate", ATYPE_FLOAT, True)
    self._num_input_channels = model_args_validator.get("num_input_channels", ATYPE_INT, True)
    self._input_size = model_args_validator.get("input_size", ATYPE_INT, True)
    self._add_biases = model_args_validator.get("add_biases", ATYPE_BOOL, True)
    self._fc_layer_dims = model_args_validator.get("fc_layer_dims", ATYPE_INTS_LIST, True)
    self._num_start_dims = model_args_validator.get("num_start_dims", ATYPE_INT, True)
    self._start_conv_size = model_args_validator.get("start_conv_size", ATYPE_INT, True)
    self._step_param = model_args_validator.get("step_param", [ATYPE_FLOAT, ATYPE_INT], True)
    self._num_units_per_group = model_args_validator.get("num_units_per_group", ATYPE_INTS_LIST, True)
    self._bottleneck_rate = model_args_validator.get("bottleneck_rate", ATYPE_FLOAT, True)




  def _get_input_var_names(self, **kwargs):
    return ["img"]





  def _get_target_var_names(self, **kwargs):
    return ["predictions"]





  def _build_cost_targets(self, in_vars, target_vars, out_vars, **kwargs):

    cost_targets = []
    cost_targets.append((target_vars["predictions"], out_vars["predictions"], None))

    return cost_targets






  def _build_prediction_network(self, input_vars, is_training, **kwargs):

    weight_vars = []
    weight_init_tups = []



    N = int(np.sum(self._num_units_per_group))
    step_factor = int(float(self._step_param) / N)
    


    # Group 1.
    cur_size = self._start_conv_size
    prev_dims = self._num_input_channels
    cur_dims = self._num_start_dims
    img_var = tf.reshape(input_vars["img"], (-1, self._input_size, self._input_size,
                                             self._num_input_channels))
    h_var = self._add_op_square_conv2d(img_var, "conv0", weight_vars,
                                       weight_init_tups, self._add_biases,
                                       prev_dims, cur_dims, cur_size)
    all_dims = [cur_dims]

    h_size = self._input_size



    # Groups with residual connections.
    for group_ind in range(len(self._num_units_per_group)):
      for i in range(self._num_units_per_group[group_ind]):

        if i == 0:
          stride = 2  # Downsample at first convolution of every group.
          add_res = False
          h_size = int(h_size // 2)
        else:
          stride = 1
          add_res = True

        prev_dims = all_dims[-1]
        cur_dims = prev_dims + step_factor
        bottleneck_dims = int(cur_dims * self._bottleneck_rate)
        k = len(all_dims)
        all_dims.append(cur_dims)


        h_res_var = h_var
        h_var = self._add_op_batch_norm(h_var, "norm_0_%d" % k, 3, is_training)

        h_var = self._add_op_square_conv2d(h_var, "conv_0_%d" % k, weight_vars,
                                           weight_init_tups, self._add_biases,
                                           prev_dims, bottleneck_dims, 1,
                                           stride=stride)

        h_var = self._add_op_batch_norm(h_var, "norm_1_%d" % k, 3, is_training)

        h_var = self._add_op_relu(h_var, "relu_0_%d" % k, alpha=self._alpha,
                                  is_variable=self._learn_alpha)

        h_var = self._add_op_square_conv2d(h_var, "conv_1_%d" % k, weight_vars,
                                           weight_init_tups, self._add_biases,
                                           bottleneck_dims, bottleneck_dims, 3)

        h_var = self._add_op_batch_norm(h_var, "norm_2_%d" % k, 3, is_training)

        h_var = self._add_op_relu(h_var, "relu_1_%d" % k, alpha=self._alpha,
                                  is_variable=self._learn_alpha)

        h_var = self._add_op_square_conv2d(h_var, "conv_2_%d" % k, weight_vars,
                                           weight_init_tups, self._add_biases,
                                           bottleneck_dims, cur_dims, 1)

        h_var = self._add_op_batch_norm(h_var, "norm_3_%d" % k, 3, is_training)

        if add_res:
          num_zeros = cur_dims - prev_dims
          paddings = np.zeros((4, 2), dtype=int)
          paddings[3, 1] = num_zeros
          h_var = h_var + tf.pad(h_res_var, paddings)





    # Fully connected and output layers.
    h_var = self._add_op_batch_norm(h_var, "norm_out", 3, is_training)
    prev_dims = all_dims[-1]
    
    for i, cur_dims in enumerate(self._fc_layer_dims + [1]):

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
        h_var = tf.nn.sigmoid(h_var)


    out_vars = {}
    out_vars["predictions"] = h_var


    return out_vars, weight_vars, weight_init_tups



