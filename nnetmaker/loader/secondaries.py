# -*- coding: utf-8 -*-
"""
Defines objects used for constructing additional tensors loaded by a batch
loader, possibly referencing tensors already loaded from data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


from nnetmaker.util import *






def _parse_shape_and_dtype(args_dict, shapes_dict, dtypes_dict):
  """Parses the shape and dtype specified in `args_dict` and checks whether they
  refer to existing tensors, and returns the dictionary key to use if so.
  Otherwise parses and returns literal shape and dtype."""


  args_val = ArgumentsValidator(args_dict, "Secondary feature")
  with args_val:
    shape = args_val.get("shape", [ATYPE_STRING, ATYPE_INTS_LIST], True)
    dtype = args_val.get("dtype", ATYPE_STRING, True)


  try:
    shape_key = shape
    static_shape = shapes_dict[shape_key]
  except KeyError:
    shape_key = None

    parsed_shape = shape
    if (shape[0] == "[" and shape[-1] == "]") or (shape[0] == "(" and shape[-1] == ")"):
      parsed_shape = shape[1:-1]

    try:
      static_shape = [int(x.strip()) for x in parsed_shape.split(",")]
    except:
      raise ValueError("Invalid shape: %s" % shape)


  try:
    dtype = dtypes_dict[dtype]
  except KeyError:
    try:
      dtype = tf.as_dtype(dtype)
    except TypeError:
      raise ValueError("Invalid dtype: %s" % dtype)


  return shape_key, static_shape, dtype








class OnesFeature(object):
  """Secondary feature for returning a tensor of ones."""

  def __init__(self, args_dict, shapes_dict, dtypes_dict):

    shape_key, static_shape, dtype = _parse_shape_and_dtype(args_dict, shapes_dict, dtypes_dict)
    self._shape_key = shape_key
    self._static_shape = static_shape
    self._dtype = dtype


  def get_static_shape(self):
    return self._static_shape

  def get_dtype(self):
    return self._dtype

  def get_tensor(self, primary_tensors_dict):
    
    if self._shape_key is not None:
      with tf.control_dependencies([primary_tensors_dict[self._shape_key]]):
        tensor = tf.ones_like(primary_tensors_dict[self._shape_key], dtype=self._dtype)

    else:
      tensor = tf.ones(self._static_shape, dtype=self._dtype)

    return tensor




