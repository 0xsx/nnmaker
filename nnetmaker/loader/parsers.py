# -*- coding: utf-8 -*-
"""
Defines parser classes used for building TFRecord parsing functions used by the
batch loader.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

import nnetmaker as nn
from nnetmaker.util import *



class RawParser(object):
  """Parser for single or sequence of tensors given as raw byte strings."""

  def __init__(self, shape, dtype, args_dict, var_len):

    self._shape = shape
    self._dtype = tf.as_dtype(dtype)
    self._var_len = var_len

    parser_val = ArgumentsValidator(args_dict, "Dataset deserializer")
    with parser_val:

      endian = parser_val.get("endian", ATYPE_STRING, True)
      if endian == "little":
        self._little_endian = True
      elif endian == "big":
        self._little_endian = False
      else:
        raise ValueError("Unsupported endianness: %s" % endian)


      if var_len:

        feat_len = parser_val.get("len", [ATYPE_NONE, ATYPE_INT], False)
        if feat_len is not None:
          raise ValueError("Length not supported for variable length feature.")

        self._extra_dim = []
        self._feat = tf.FixedLenSequenceFeature([], tf.string)

      else:
        feat_len = parser_val.get("len", ATYPE_INT, False, default=1)

        if feat_len > 1:
          self._extra_dim = [feat_len]
        else:
          self._extra_dim = []

        self._feat = tf.FixedLenFeature(self._extra_dim, tf.string)



  def get_feat(self):
    return self._feat, self._var_len


  def parse(self, in_tensor):
    tensor = tf.decode_raw(in_tensor, self._dtype, little_endian=self._little_endian)
    tensor = tf.reshape(tensor, self._extra_dim + self._shape)
    return tensor


  def parse_multi(self, in_tensor, batch_size):
    tensor = tf.decode_raw(in_tensor, self._dtype, little_endian=self._little_endian)
    tensor = tf.reshape(tensor, [batch_size] + self._extra_dim + self._shape)
    return tensor







class Int64Parser(object):
  """Parser for a tensor given as an int64 list."""

  def __init__(self, shape, dtype, args_dict, var_len):

    self._dtype = tf.as_dtype(dtype)
    self._var_len = var_len

    if len(args_dict) > 0:
      raise ValueError("Arguments not supported for int deserialization.")

    if var_len:
      self._feat = tf.FixedLenSequenceFeature(shape, tf.int64)
    else:
      self._feat = tf.FixedLenFeature(shape, tf.int64)


  def get_feat(self):
    return self._feat, self._var_len


  def parse(self, in_tensor):
    if self._dtype == tf.int64:
      tensor = in_tensor
    else:
      tensor = tf.cast(in_tensor, self._dtype)
    return tensor


  def parse_multi(self, in_tensor, batch_size):
    return self.parse(in_tensor)







class FloatParser(object):
  """Parser for a tensor given as a float list."""

  def __init__(self, shape, dtype, args_dict, var_len):

    self._dtype = tf.as_dtype(dtype)
    self._var_len = var_len

    if len(args_dict) > 0:
      raise ValueError("Arguments not supported for float deserialization.")

    if var_len:
      self._feat = tf.FixedLenSequenceFeature(shape, tf.float32)
    else:
      self._feat = tf.FixedLenFeature(shape, tf.float32)


  def get_feat(self):
    return self._feat, self._var_len


  def parse(self, in_tensor):
    if self._dtype == tf.float32:
      tensor = in_tensor
    else:
      tensor = tf.cast(in_tensor, self._dtype)
    return tensor


  def parse_multi(self, in_tensor, batch_size):
    return self.parse(in_tensor)









class StringParser(object):
  """Parser for a tensor of strings given as a bytes list."""

  def __init__(self, shape, dtype, args_dict, var_len):

    self._dtype = tf.as_dtype(dtype)
    self._var_len = var_len

    if len(args_dict) > 0:
      raise ValueError("Arguments not supported for string deserialization.")

    if var_len:
      self._feat = tf.FixedLenSequenceFeature(shape, tf.string)
    else:
      self._feat = tf.FixedLenFeature(shape, tf.string)


  def get_feat(self):
    return self._feat, self._var_len


  def parse(self, in_tensor):
    if self._dtype == tf.string:
      tensor = in_tensor
    else:
      tensor = tf.cast(in_tensor, self._dtype)
    return tensor


  def parse_multi(self, in_tensor, batch_size):
    return self.parse(in_tensor)







