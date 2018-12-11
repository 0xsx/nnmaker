# -*- coding: utf-8 -*-
"""
Defines the base class for constructing models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np
import os
import tensorflow as tf


from nnetmaker.util import *


FLOAT_DTYPE = "float32"




class BaseModel(object):
  """Base class for prediction models."""


  @property
  def input_names(self):
    """A list of names of model input variables."""
    return self._get_input_var_names()


  @property
  def target_names(self):
    """A list of names of model target variables."""
    return self._get_target_var_names()


  @property
  def model_dir(self):
    """The directory for saving the Estimator model weights."""
    return self._model_dir


  @property
  def model_name(self):
    """The configured name of the model."""
    return self._model_name






  def __init__(self, model_args_dict, epsilon=float(1e-6), **kwargs):
    """Initializes a new model. Must be called by all subclasses."""

    # Subclasses must be careful not to overwrite these members, or any
    # methods that are not reserved for subclasses

    self._epsilon = epsilon
    self._batch_size = 1

    model_args_validator = ArgumentsValidator(model_args_dict, "Model arguments")
    with model_args_validator:
      self._model_name = model_args_validator.get("name", ATYPE_STRING, True)
      self._model_dir = os.path.realpath(model_args_validator.get("dir", ATYPE_STRING, True))
      self._process_args(model_args_validator, **kwargs)
    
    




  def build_weight_init_tups(self, features, labels, params, **kwargs):
    """Builds the training graph and returns the list of weight initialization
    tuples."""

    with tf.name_scope(None):
      tup = self._build_prediction_network(features, False, **kwargs)
      out_vars = tup[0]
      weight_vars = tup[1]
      weight_init_tups = tup[2]

      obj_cost_var = self._build_cost(features, labels, out_vars, params, **kwargs)
      train_step = self._build_train_step(obj_cost_var, weight_vars, params)
      self._build_metrics(features, labels, out_vars, **kwargs)

      tf.summary.tensor_summary("obj_cost", obj_cost_var)


    return weight_init_tups





  def build_predict_graph(self, placeholders, **kwargs):
    """Builds the prediction graph and returns a dict of input tensor names, a
    dict of output tensor names, and a list of output op names."""

    with tf.name_scope(None):
      out_vars, _, _ = self._build_prediction_network(placeholders, False, **kwargs)


    in_tensor_names = {}
    out_tensor_names = {}
    out_op_names = []

    for name in placeholders:
      in_tensor_names[name] = placeholders[name].name

    for name in out_vars:
      out_tensor_names[name] = out_vars[name].name
      out_op_names.append(out_vars[name].op.name)


    return in_tensor_names, out_tensor_names, out_op_names




  def build_model_fn(self, **kwargs):
    """Constructs the model function for creating a new Tensorflow
    Estimator."""


    def model_fn(features, labels, mode, params, **kwargs):

      with tf.name_scope(None):

        if mode == tf.estimator.ModeKeys.TRAIN:
          out_vars, weight_vars, _ = self._build_prediction_network(features, True, **kwargs)
          obj_cost_var = self._build_cost(features, labels, out_vars, params, **kwargs)
          train_step = self._build_train_step(obj_cost_var, weight_vars, params)

          return tf.estimator.EstimatorSpec(mode, loss=obj_cost_var,
                                            train_op=train_step)


        elif mode == tf.estimator.ModeKeys.EVAL:
          out_vars, _, _ = self._build_prediction_network(features, False, **kwargs)
          obj_cost_var = self._build_cost(features, labels, out_vars, params, **kwargs)

          metrics_dict = self._build_metrics(features, labels, out_vars, **kwargs)
          for name in metrics_dict:
            tf.summary.scalar(name, metrics_dict[name][0])
          tf.summary.scalar("loss", obj_cost_var)

          return tf.estimator.EstimatorSpec(mode, loss=obj_cost_var,
                                            eval_metric_ops=metrics_dict)


        elif mode == tf.estimator.ModeKeys.PREDICT:
          out_vars, _, _ = self._build_prediction_network(features, False, **kwargs)
          return tf.estimator.EstimatorSpec(mode, predictions=out_vars)

        else:
          raise ValueError("Unrecognized model mode.")


    return model_fn







  def _build_cost(self, in_vars, target_vars, out_vars, params, **kwargs):
    """Constructs the unregularized objective cost."""

    cost_targets = self._build_cost_targets(in_vars, target_vars, out_vars, **kwargs)

    objective_cost_var = None

    cost_type = params["cost_type"]

    for target_var, out_var, mask_var in cost_targets:

      if cost_type == "nll":
        # Negative log likelihood.
        clipped_out_var = tf.clip_by_value(out_var, self._epsilon, 1 - self._epsilon)
        all_costs_var = -tf.log(clipped_out_var) * target_var
        if mask_var is not None:
          all_costs_var *= mask_var
        cur_cost_var = tf.reduce_sum(all_costs_var)


      elif cost_type == "xent":
        # Cross entropy.
        clipped_out_var = tf.clip_by_value(out_var, self._epsilon, 1 - self._epsilon)
        all_costs_var = -(target_var * tf.log(clipped_out_var)
                          + (1 - target_var) * tf.log(1 - clipped_out_var))
        if mask_var is not None:
          all_costs_var *= mask_var
        cur_cost_var = tf.reduce_sum(all_costs_var)


        #### TODO
        # tf.nn.sparse_softmax_cross_entropy_with_logits


      elif cost_type == "sse":
        # Sum of squares error.
        all_costs_var = (target_var - out_var)**2
        if mask_var is not None:
          all_costs_var *= mask_var
        cur_cost_var = tf.reduce_sum(all_costs_var)

      else:
        raise ValueError("Unrecognized cost type: %s" % cost_type)


      if objective_cost_var is None:
        objective_cost_var = cur_cost_var
      else:
        objective_cost_var += cur_cost_var


    assert objective_cost_var is not None


    return objective_cost_var








  def _build_train_step(self, obj_cost_var, weight_vars, params):
    """Constructs the training step from the regularized objective cost."""

    reg_weight_decay = params["reg_weight_decay"]
    max_grad_norm = params["max_grad_norm"]
    parallel_grad_gate = params["parallel_grad_gate"]

    use_rmsprop = params["use_rmsprop"]
    if use_rmsprop:
      rmsprop_momentum = params["rmsprop_momentum"]
      rmsprop_decay = params["rmsprop_decay"]
      rmsprop_epsilon = params["rmsprop_epsilon"]
    else:
      sgd_momentum = params["sgd_momentum"]
      sgd_use_nesterov = params["sgd_use_nesterov"]

    lr = params["learning_rate"]


    global_step_var = tf.train.get_or_create_global_step()


    # Apply l2 regularization to weights.
    if reg_weight_decay is not None:
      l2_losses = [tf.nn.l2_loss(tf.cast(w, FLOAT_DTYPE)) for w in weight_vars]
      l2_loss_var = reg_weight_decay * tf.add_n(l2_losses)
      obj_cost_var = obj_cost_var + l2_loss_var


    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):

      if use_rmsprop:
        opt_op = tf.train.RMSPropOptimizer(lr, momentum=rmsprop_momentum,
                                           decay=rmsprop_decay, epsilon=rmsprop_epsilon)
      else:
        if sgd_momentum is not None:
          opt_op = tf.train.MomentumOptimizer(lr, sgd_momentum,
                                              use_nesterov=sgd_use_nesterov)
        else:
          opt_op = tf.train.GradientDescentOptimizer(lr)
        


      if parallel_grad_gate:
        gate = tf.train.Optimizer.GATE_NONE
      else:
        gate = tf.train.Optimizer.GATE_GRAPH

      if max_grad_norm is not None:
        # Enable gradient clipping.
        grads_list = opt_op.compute_gradients(obj_cost_var, gate_gradients=gate)
        gradients = []
        variables = []
        for g, v in grads_list:
          if g is not None and v is not None:
            gradients.append(g)
            variables.append(v)
        casted_gradients = [tf.cast(g, FLOAT_DTYPE) for g in gradients]
        clipped_gradients, _ = tf.clip_by_global_norm(casted_gradients, max_grad_norm)
        gradients = [tf.cast(clipped_gradients[i],
                             variables[i].dtype) for i in range(len(clipped_gradients))]
        opt_step = opt_op.apply_gradients(zip(gradients, variables),
                                          global_step=global_step_var)

      else:
        opt_step = opt_op.minimize(obj_cost_var, gate_gradients=gate,
                                   global_step=global_step_var)


    return opt_step





  #############################################################
  # GRAPH OP METHODS
  #############################################################



  def _add_op_square_conv2d(self, in_var, name, weight_vars,
                            weight_init_tups, add_bias, num_in_channels,
                            num_out_channels, size, stride=1, dilation=1,
                            pad=True):
    """Creates a 2d convolution op in the graph to convolve the given input
    with a square filter. Input volume must have the shape
    `(batch_size, width, height, channels)`."""


    w_shape = (size, size, num_in_channels, num_out_channels)
    b_shape = (num_out_channels,)

    # The shape for assignment has output channels along the first dimension
    # to make sure output channels are initialized orthogonal to each
    # other by the initialization program.
    w_assign_shape = (num_out_channels, size, size, num_in_channels)


    with tf.variable_scope(name):

      w_var = tf.get_variable("w", shape=w_shape, dtype=FLOAT_DTYPE,
                              trainable=True, initializer=tf.zeros_initializer())
      weight_vars.append(w_var)


      def w_assign_fn(w_arr, tf_session):
        # Need to swap output channels to the last dimension for TensorFlow.
        w_arr = np.transpose(w_arr, (1, 2, 3, 0))
        tf_session.run(tf.assign(w_var, w_arr))
      
      



      strides = [1, stride, stride, 1]
      dilations = [1, dilation, dilation, 1]

      if pad:
        pad_mode = "SAME"
      else:
        pad_mode = "VALID"

      out_var = tf.nn.conv2d(in_var, w_var, strides, pad_mode,
                             dilations=dilations, data_format="NHWC")

      if add_bias:
        b_var = tf.get_variable("b", shape=b_shape, dtype=FLOAT_DTYPE,
                                trainable=True, initializer=tf.zeros_initializer())
        out_var = out_var + b_var[None, None, None, :]


    allow_lsuv_init = True
    weight_init_tups.append((w_assign_shape, w_assign_fn, out_var,
                             allow_lsuv_init, out_var.name,
                             out_var.dtype.as_numpy_dtype))

    
    return out_var




  def _add_op_batch_norm(self, in_var, name, axis, is_training):
    """Creates a batch normalization op in the graph to normalize `in_var`
    across the specified axis. Returns the normalized variable."""

    out_var = tf.layers.batch_normalization(in_var, training=is_training,
                                            trainable=True, axis=axis, name=name)

    return out_var




  def _add_op_relu(self, in_var, name, alpha=0., is_variable=False):
    """Creates a ReLU op in the graph. If `alpha` is nonzero, leaky ReLU is
    performed. If `is_variable` is True, a variable will be created for the
    alpha value to be learned. Returns the post-activation variable."""


    with tf.variable_scope(name):

      if alpha > 0:
        if is_variable:
          alpha_var = tf.get_variable("alpha", shape=[], dtype=FLOAT_DTYPE,
                                      trainable=True,
                                      initializer=tf.constant_initializer(alpha))
          out_var = tf.nn.leaky_relu(in_var, alpha=alpha_var)
        else:
          alpha_const = tf.constant(alpha, dtype=FLOAT_DTYPE)
          out_var = tf.nn.leaky_relu(in_var, alpha=alpha_const)

      else:
        out_var = tf.nn.relu(in_var)


    return out_var




  def _add_op_dropout(self, in_var, name, rate, is_training, noise_shape=None):
    """Creates a dropout op in the graph to drop `rate` percentage of
    values from `in_var` if training is enabled. Has no effect if
    training is not enabled. Returns the output variable."""

    out_var = tf.layers.dropout(in_var, rate=rate, noise_shape=noise_shape,
                                training=is_training, name=name)


    return out_var








  #############################################################
  # SUBCLASS METHODS
  #############################################################

  def _process_args(self, model_args_validator, **kwargs):
    """Must be implemented by subclasses to process model arguments using the
    specified validator object."""
    raise NotImplementedError



  def _get_input_var_names(self, **kwargs):
    """Must be implemented by subclasses to return a list of names of
    input variables."""
    raise NotImplementedError



  def _get_target_var_names(self, **kwargs):
    """Must be implemented by subclasses to return a list of names of
    target output variables used to compute the objective function."""
    raise NotImplementedError



  def _build_prediction_network(self, input_vars, is_training, **kwargs):
    """Must be implemented by subclasses to construct the part of the network
    that produces output predictions. All variables are under the scope
    `self._model_name`. Variables must have default initializers set and must
    have unique names/scopes.

    Implementations must return a tuple:
    `(out_vars, weight_vars, weight_init_tups)`

      `out_vars` - dictionary of predictions computed from the current inputs,
                   with the names specified as outputs; first dimension must
                   match the batch size
      `weight_vars` - a list of all learned weight variables in the network
      `weight_init_tups` - a list of tuples used for initializing weight
                           parameters, one tuple per weight, in the order
                           encountered in the network

    Regularization will be applied by the training network. Only weight
    variables will be regularized. Alpha vars should not be
    regularized per He et al. 2015: https://arxiv.org/abs/1502.01852
    """
    raise NotImplementedError



  def _build_cost_targets(self, in_vars, target_vars, out_vars, **kwargs):
    """Must be implemented by subclasses to return a list of tuples used for
    computing the cost. Inputs are dictionary objects.

    Each tuple must be of the form:
    `(target_var, out_var, mask_var)`

      `target_var` - the tensor storing ground truth target values
      `out_var` - the tensor predicted by the network
      `mask_var` - either a binary tensor or `None` to indicate no masking
    """
    raise NotImplementedError



  def _build_metrics(self, in_vars, target_vars, out_vars, **kwargs):
    """May be implemented by subclasses to return any metrics tracked by
    TensorFlow. Must return a dictionary mapping metric names to tuples of the
    form `(metric_tensor, update_op)` as returned by metric functions."""
    return {}


