# -*- coding: utf-8 -*-
"""
Defines utility constants, classes, methods.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


__all__ = ["ATYPE_NONE", "ATYPE_STRING", "ATYPE_INT", "ATYPE_FLOAT",
           "ATYPE_BOOL", "ATYPE_DICT", "ATYPE_LIST", "ATYPE_STRINGS_LIST",
           "ATYPE_INTS_LIST", "ATYPE_FLOATS_LIST", "ATYPE_BOOLS_LIST",
           "ATYPE_DICTS_LIST", "ATYPE_LISTS_LIST", "ArgumentMissingError",
           "ArgumentTypeError","ArgumentUnexpectedError", "ArgumentsValidator",
           "read_json_object"]


import io
import json


# Available data types for arguments.
ATYPE_NONE = "null"
ATYPE_STRING = "string"
ATYPE_INT = "integer"
ATYPE_FLOAT = "float"
ATYPE_BOOL = "boolean"
ATYPE_DICT = "dictionary"
ATYPE_LIST = "list"
ATYPE_STRINGS_LIST = "list of strings"
ATYPE_INTS_LIST = "list of integers"
ATYPE_FLOATS_LIST = "list of floats"
ATYPE_BOOLS_LIST = "list of booleans"
ATYPE_DICTS_LIST = "list of dictionaries"
ATYPE_LISTS_LIST = "list of lists"



# If numeric conversion changes a number value by this amount or more,
# consider it an error.
_DIFF_EPSILON = float(1e-6)




# Methods for validating each type defined above. Each method returns a
# tuple where the first item is a boolean indicating whether the specified
# value is given as the requested type, and the second item is the
# value converted to that type if the first item is ``True``. If the first
# item is ``False``, the second item is the original value. Strings are not
# converted to numeric values or vice-versa.

def _val_none(val):
  if val is None:
    return True, None
  return False, val

def _val_string(val):
  try:
    return isinstance(val, (str, unicode)), val
  except NameError:
    return isinstance(val, str), val

def _val_bool(val):
  return isinstance(val, bool), val

def _val_int(val):
  if isinstance(val, bool):
    return False, val
  if isinstance(val, float):
    if abs(int(val) - val) < _DIFF_EPSILON:
      return True, int(val)
    return False, val
  try:
    return isinstance(val, (int, long)), val
  except NameError:
    return isinstance(val, int), val

def _val_float(val):
  try:
    if isinstance(val, (int, long, float)) and not isinstance(val, bool):
      return True, float(val)
  except NameError:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
      return True, float(val)
  return False, val

def _val_dict(val):
  return isinstance(val, dict), val

def _val_list(val):
  if isinstance(val, (list, tuple)):
    return True, list(val)
  return False, val

def _val_list_of(val, inner_val_fn):
  is_list, list_val = _val_list(val)
  if not is_list: return False, val
  ret_val = []
  for item in list_val:
    is_type, item = inner_val_fn(item)
    if not is_type: return False, val
    ret_val.append(item)
  return True, ret_val

def _val_strings_list(val):
  return _val_list_of(val, _val_string)

def _val_ints_list(val):
  return _val_list_of(val, _val_int)

def _val_floats_list(val):
  return _val_list_of(val, _val_float)

def _val_bools_list(val):
  return _val_list_of(val, _val_bool)

def _val_dicts_list(val):
  return _val_list_of(val, _val_dict)

def _val_lists_list(val):
  return _val_list_of(val, _val_list)


_VAL_FNS = {
    ATYPE_NONE: _val_none,
    ATYPE_STRING: _val_string,
    ATYPE_INT: _val_int,
    ATYPE_FLOAT: _val_float,
    ATYPE_BOOL: _val_bool,
    ATYPE_DICT: _val_dict,
    ATYPE_LIST: _val_list,
    ATYPE_STRINGS_LIST: _val_strings_list,
    ATYPE_INTS_LIST: _val_ints_list,
    ATYPE_FLOATS_LIST: _val_floats_list,
    ATYPE_BOOLS_LIST: _val_bools_list,
    ATYPE_DICTS_LIST: _val_dicts_list,
    ATYPE_LISTS_LIST: _val_lists_list
}






def read_json_object(json_filename, encoding="utf-8"):
  """Reads and returns a single JSON object from the specified text file,
  raising an error if the file cannot be parsed or if it does not contain
  exactly one JSON object as a dictionary."""

  # Strip comments while keeping line and column numbers.
  stripped_str = ""
  with io.open(json_filename, "r", encoding=encoding) as f_in:
    for line in f_in:
      line = line.rstrip()
      comment_pos = line.find("//")
      if comment_pos > -1:
        line = line[:comment_pos]
      stripped_str += line + "\n"

  obj = json.loads(stripped_str)

  if not isinstance(obj, dict):
    raise ValueError("File %s contains an invalid JSON object." % json_filename)

  return obj






class ArgumentMissingError(Exception):
  """Raised by the arguments validator when an argument is missing."""

  def __init__(self, arg_name, tag):
    self.msg = "%s - missing required argument `%s`." % (tag, arg_name)
    self.arg_name = arg_name
    self.tag = tag

  def __str__(self):
    return self.msg




class ArgumentTypeError(Exception):
  """Raised by the arguments validator when an argument has unexpected type."""

  def __init__(self, arg_name, tag, expected_types):
    self.msg = "%s - argument `%s` has invalid type. " % (tag, arg_name)

    if isinstance(expected_types, str):
      self.msg += "Expected %s." % expected_types
    else:
      self.msg += "Expected one of: %s." % ", ".join(expected_types)

    self.arg_name = arg_name
    self.tag = tag

  def __str__(self):
    return self.msg




class ArgumentUnexpectedError(Exception):
  """Raised by the arguments validator when an unexpected argument is given."""

  def __init__(self, arg_name, tag):
    self.msg = "%s - got unexpected argument `%s`." % (tag, arg_name)
    self.arg_name = arg_name
    self.tag = tag

  def __str__(self):
    return self.msg







class ArgumentsValidator(object):
  """Retrieves arguments from a specified dictionary, raising errors if the
  data types or structure do not match what is expected. Not multiprocessing
  safe."""


  def __init__(self, args_dict, tag):
    """Initializes a validator for the specified dictionary of arguments and a
    string ``tag`` that is prepended to any error messages."""

    if not isinstance(args_dict, dict):
      raise ValueError("Expected dictionary of arguments.")

    self._args_dict = args_dict
    self._tag = tag
    self._tracked_arg_accesses = None

    


  def get(self, arg_name, arg_type, is_required, default=None):
    """Gets the value of the specified argument if it exists in the arguments
    dictionary provided at construction. If the value does not match the
    one of the specified types, an ``ArgumentTypeError`` is raised.
    ``arg_type`` may be a single type or an iterable of possible types. If the
    argument is missing and ``is_required`` is set, an ``ArgumentMissingError``
    will be raised. If the argument is missing and not required, the default is
    returned instead."""

    try:
      arg_val = self._args_dict[arg_name]
    except KeyError:
      if is_required:
        raise ArgumentMissingError(arg_name, self._tag)
      arg_val = default

    types_list = []
    if isinstance(arg_type, str):
      types_list.append(arg_type)
    else:
      types_list.extend(arg_type)


    # Make sure specified default value is actually allowed if argument is
    # not required.
    if not is_required:
      is_valid = False
      for t in types_list:
        is_valid, _ = _VAL_FNS[t](default)
        if is_valid: break
      if not is_valid:
        raise ValueError("Specified default argument does not match allowed type.")


    # Make sure argument value we want to return is a valid type, or raise
    # an error if not.
    is_valid = False
    for t in types_list:
      is_valid, ret_val = _VAL_FNS[t](arg_val)
      if is_valid: break
    if not is_valid:
        raise ArgumentTypeError(arg_name, self._tag, arg_type)


    # If we are tracking argument accesses with a context manager, update the
    # set of accessed arguments.
    if self._tracked_arg_accesses is not None:
      self._tracked_arg_accesses.add(arg_name)


    return ret_val




  def __enter__(self):
    """Enters a context that begins tracking the set of arguments accessed."""

    if self._tracked_arg_accesses is not None:
      raise RuntimeError("Context already entered.")
    self._tracked_arg_accesses = set()




  def __exit__(self, exc_type, exc_val, exc_tb):
    """Exits the context and checks whether all arguments in the dictionary
    were accessed. If an argument is found that has not been accessed since the
    context was entered, an ``ArgumentUnexpectedError`` is raised."""

    if self._tracked_arg_accesses is None:
      raise RuntimeError("Context not entered.")

    if exc_type is None and exc_val is None and exc_tb is None:
      unaccessed = set(self._args_dict.keys()) - self._tracked_arg_accesses
      if unaccessed:
        first_unaccessed = sorted(unaccessed)[0]
        raise ArgumentUnexpectedError(first_unaccessed, self._tag)
      self._tracked_arg_accesses = None


    return False



