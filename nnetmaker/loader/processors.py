# -*- coding: utf-8 -*-
"""
Defines classes for preprocessing data tensors after they have been
parsed from TFRecords.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from nnetmaker.util import *







def _parse_slice_str(slice_str):
  """Parses the given string as a multidimensional array slice and returns a
  list of slice objects and integer indices."""

  is_valid = False
  if len(slice_str) > 2:
    is_valid = slice_str[0] == "[" and slice_str[-1] == "]"


  sliced_inds = []
  if is_valid:
    slice_str_list = [x.strip() for x in slice_str[1:-1].split(",")]

    for s in slice_str_list:
      parts = s.split(":")
      if len(parts) > 3:
        is_valid = False
        break

      if len(parts) == 1:
        try:
          sliced_inds.append(int(s))
        except:
          is_valid = False
          break

      else:
        try:
          start = int(parts[0]) if len(parts[0]) > 0 else None
          stop = int(parts[1]) if len(parts[1]) > 0 else None
          if len(parts) == 3:
            step = int(parts[2]) if len(parts[2]) > 0 else None
          else:
            step = None

        except:
          is_valid = False
          break

        sliced_inds.append(slice(start, stop, step))


  if not is_valid:
    raise ValueError("Invalid slice specified: %s" % slice_str)


  return sliced_inds






class SliceProcessor(object):
  """Processor for extracting a slice from a tensor."""

  def __init__(self, args_dict, in_shape, in_dtype):

    args_val = ArgumentsValidator(args_dict, "Slice processor")
    with args_val:
      slice_str = args_val.get("slice", ATYPE_STRING, True)
    sliced_inds = _parse_slice_str(slice_str)

    if len(sliced_inds) > len(in_shape):
      raise ValueError("Too many dimensions for slice. Expected %d, got %d."
                       % (len(in_shape), len(sliced_inds)))

    out_shape = []
    for obj in sliced_inds:
      if isinstance(obj, slice):
        out_shape.append(-1)     # Any integer indices remove a dimension; only
                                 # sliced dimensions remain.

    out_shape += in_shape[len(sliced_inds):]


    if len(sliced_inds) == 1:
      self._sliced_inds = sliced_inds[0]
    else:
      self._sliced_inds = sliced_inds


    self._dtype = in_dtype
    self._static_shape = out_shape


  def get_static_shape(self):
    return self._static_shape

  def get_dtype(self):
    return self._dtype

  def process(self, in_tensor):
    return in_tensor.__getitem__(self._sliced_inds)







class OneHotProcessor(object):
  """Processor for expanding an integer tensor to a float vector with one
  element set to 1 and all others set to 0."""

  def __init__(self, args_dict, in_shape, in_dtype):

    if in_dtype.name != "int32" and in_dtype.name != "int64":
      raise ValueError("Invalid datatype for one hot indices.")

    args_val = ArgumentsValidator(args_dict, "One Hot processor")

    with args_val:
      self._depth = args_val.get("depth", ATYPE_INT, True)
      dtype_str = args_val.get("dtype", ATYPE_STRING, False, default="float32")
      self._dtype = tf.as_dtype(dtype_str)

    self._static_shape = in_shape + [self._depth]


  def get_static_shape(self):
    return self._static_shape

  def get_dtype(self):
    return self._dtype

  def process(self, in_tensor):
    return tf.one_hot(in_tensor, self._depth, dtype=self._dtype)







class ReshapeProcessor(object):
  """Processor for reshaping a tensor."""

  def __init__(self, args_dict, in_shape, in_dtype):

    args_val = ArgumentsValidator(args_dict, "Reshape processor")

    with args_val:
      self._static_shape = args_val.get("shape", ATYPE_INTS_LIST, True)

    for d in self._static_shape:
      if d <= 0:
        raise ValueError("Non-positive dimensions not supported.")

    self._dtype = in_dtype


  def get_static_shape(self):
    return self._static_shape

  def get_dtype(self):
    return self._dtype

  def process(self, in_tensor):
    return tf.reshape(in_tensor, self._static_shape)









