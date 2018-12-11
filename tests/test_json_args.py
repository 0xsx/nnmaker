#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines test cases for loading and validating arguments from JSON files.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import os
import unittest

from test_import import TEST_DATA_PATH
from test_import import nn





class JsonArgsTestCase(unittest.TestCase):

  def setUp(self):
    self.longMessage = True
    self.data_path = os.path.join(TEST_DATA_PATH, "test_json_args")



  def test_json_syntax_error_lineno(self):
    """Test decoding error sees correct line number on syntax error."""
    lineno = None
    filename = os.path.join(self.data_path, "invalid_syntax.json")

    try:
      try:
        obj = nn.read_json_object(filename)
      except json.decoder.JSONDecodeError as e:
        lineno = e.lineno
      self.assertEqual(lineno, 13, "invalid syntax error line number")

    except AttributeError:  # python 2
      with self.assertRaises(ValueError):
        obj = nn.read_json_object(filename)

    


  def test_json_syntax_error_colno(self):
    """Test decoding error sees correct column number on syntax error."""
    colno = None
    filename = os.path.join(self.data_path, "invalid_syntax.json")

    try:
      try:
        obj = nn.read_json_object(filename)
      except json.decoder.JSONDecodeError as e:
        colno = e.colno

      self.assertEqual(colno, 13, "invalid syntax error column number")

    except AttributeError:  # python 2
      with self.assertRaises(ValueError):
        obj = nn.read_json_object(filename)



  def test_json_syntax_error_lineno_crlf(self):
    """Test decoding error sees correct line number on syntax error with CRLF
    line endings."""
    lineno = None
    filename = os.path.join(self.data_path, "invalid_syntax_crlf.json")

    try:
      try:
        obj = nn.read_json_object(filename)
      except json.decoder.JSONDecodeError as e:
        lineno = e.lineno

      self.assertEqual(lineno, 13, "invalid syntax error line number with CRLF EOL")
    except AttributeError:  # python 2
      with self.assertRaises(ValueError):
        obj = nn.read_json_object(filename)


  def test_json_syntax_error_colno_crlf(self):
    """Test decoding error sees correct column number on syntax error with CRLF
    line endings."""
    colno = None
    filename = os.path.join(self.data_path, "invalid_syntax_crlf.json")

    try:
      try:
        obj = nn.read_json_object(filename)
      except json.decoder.JSONDecodeError as e:
        colno = e.colno

      self.assertEqual(colno, 13, "invalid syntax error column number with CRLF EOL")

    except AttributeError:  # python 2
      with self.assertRaises(ValueError):
        obj = nn.read_json_object(filename)



  def test_json_non_object_error(self):
    """Test raise error on non JSON object in file."""
    colno = None
    filename = os.path.join(self.data_path, "non_object.json")

    with self.assertRaises(ValueError):
      obj = nn.read_json_object(filename)



  def test_json_multi_object_error(self):
    """Test raise error on multiple objects in file."""
    colno = None
    filename = os.path.join(self.data_path, "too_many_objects.json")

    try:
      with self.assertRaises(json.decoder.JSONDecodeError):
        obj = nn.read_json_object(filename)
    except AttributeError:  # python 2
      with self.assertRaises(ValueError):
        obj = nn.read_json_object(filename)


  def test_arg_types(self):
    """Test raise error only on invalid type specified."""
    filename = os.path.join(self.data_path, "type_checks.json")

    obj = nn.read_json_object(filename)
    val = nn.ArgumentsValidator(obj, "Args Test")

    all_types = set([nn.ATYPE_NONE, nn.ATYPE_STRING, nn.ATYPE_INT, nn.ATYPE_FLOAT,
                     nn.ATYPE_BOOL, nn.ATYPE_DICT, nn.ATYPE_LIST, nn.ATYPE_STRINGS_LIST,
                     nn.ATYPE_INTS_LIST, nn.ATYPE_FLOATS_LIST, nn.ATYPE_BOOLS_LIST,
                     nn.ATYPE_DICTS_LIST, nn.ATYPE_LISTS_LIST])

    def __test(arg_name, allowed_types):
      for t in all_types - allowed_types:
        with self.assertRaises(nn.ArgumentTypeError):
          val.get(arg_name, t, True)
      for t in allowed_types:
        val.get(arg_name, t, True)

    __test("test1", set([nn.ATYPE_STRING]))
    __test("test2", set([nn.ATYPE_STRING]))
    __test("test3", set([nn.ATYPE_INT, nn.ATYPE_FLOAT]))
    __test("test4", set([nn.ATYPE_FLOAT]))
    __test("test5", set([nn.ATYPE_LIST, nn.ATYPE_INTS_LIST, nn.ATYPE_FLOATS_LIST]))
    __test("test6", set([nn.ATYPE_LIST, nn.ATYPE_INTS_LIST, nn.ATYPE_FLOATS_LIST]))
    __test("test7", set([nn.ATYPE_LIST]))
    __test("test8", set([nn.ATYPE_LIST, nn.ATYPE_STRINGS_LIST, nn.ATYPE_INTS_LIST,
                         nn.ATYPE_FLOATS_LIST, nn.ATYPE_DICTS_LIST, nn.ATYPE_LISTS_LIST,
                         nn.ATYPE_BOOLS_LIST]))
    __test("test9", set([nn.ATYPE_LIST]))
    __test("test10", set([nn.ATYPE_LIST, nn.ATYPE_FLOATS_LIST, nn.ATYPE_INTS_LIST]))
    __test("test11", set([nn.ATYPE_LIST, nn.ATYPE_STRINGS_LIST]))
    __test("test12", set([nn.ATYPE_BOOL]))
    __test("test13", set([nn.ATYPE_INT, nn.ATYPE_FLOAT]))
    __test("test14", set([nn.ATYPE_DICT]))
    __test("test15", set([nn.ATYPE_NONE]))
    __test("test16", set([nn.ATYPE_LIST, nn.ATYPE_BOOLS_LIST]))
    __test("test17", set([nn.ATYPE_LIST, nn.ATYPE_FLOATS_LIST]))
    __test("test18", set([nn.ATYPE_LIST, nn.ATYPE_DICTS_LIST]))
    __test("test19", set([nn.ATYPE_STRING]))
    __test("test20", set([nn.ATYPE_LIST, nn.ATYPE_LISTS_LIST]))





  def test_arg_default_types(self):
    """Test raise error on invalid type for default."""
    filename = os.path.join(self.data_path, "type_checks.json")

    obj = nn.read_json_object(filename)
    val = nn.ArgumentsValidator(obj, "Args Test")

    all_types = set([nn.ATYPE_NONE, nn.ATYPE_STRING, nn.ATYPE_INT, nn.ATYPE_FLOAT,
                     nn.ATYPE_BOOL, nn.ATYPE_DICT, nn.ATYPE_LIST, nn.ATYPE_STRINGS_LIST,
                     nn.ATYPE_INTS_LIST, nn.ATYPE_FLOATS_LIST, nn.ATYPE_BOOLS_LIST,
                     nn.ATYPE_DICTS_LIST, nn.ATYPE_LISTS_LIST])

    def __test(arg_name, allowed_types):
      
      for t in allowed_types:
        v = val.get(arg_name, t, True)
        val.get(arg_name, t, False, default=v)

        if v is not None:
          with self.assertRaises(ValueError):
            val.get(arg_name, t, False, default=None)
        else:
          with self.assertRaises(ValueError):
            val.get(arg_name, t, False, default="")

    __test("test1", set([nn.ATYPE_STRING]))
    __test("test2", set([nn.ATYPE_STRING]))
    __test("test3", set([nn.ATYPE_INT, nn.ATYPE_FLOAT]))
    __test("test4", set([nn.ATYPE_FLOAT]))
    __test("test5", set([nn.ATYPE_LIST, nn.ATYPE_INTS_LIST, nn.ATYPE_FLOATS_LIST]))
    __test("test6", set([nn.ATYPE_LIST, nn.ATYPE_INTS_LIST, nn.ATYPE_FLOATS_LIST]))
    __test("test7", set([nn.ATYPE_LIST]))
    __test("test8", set([nn.ATYPE_LIST, nn.ATYPE_STRINGS_LIST, nn.ATYPE_INTS_LIST,
                         nn.ATYPE_FLOATS_LIST, nn.ATYPE_DICTS_LIST, nn.ATYPE_LISTS_LIST,
                         nn.ATYPE_BOOLS_LIST]))
    __test("test9", set([nn.ATYPE_LIST]))
    __test("test10", set([nn.ATYPE_LIST, nn.ATYPE_FLOATS_LIST, nn.ATYPE_INTS_LIST]))
    __test("test11", set([nn.ATYPE_LIST, nn.ATYPE_STRINGS_LIST]))
    __test("test12", set([nn.ATYPE_BOOL]))
    __test("test13", set([nn.ATYPE_INT, nn.ATYPE_FLOAT]))
    __test("test14", set([nn.ATYPE_DICT]))
    __test("test15", set([nn.ATYPE_NONE]))
    __test("test16", set([nn.ATYPE_LIST, nn.ATYPE_BOOLS_LIST]))
    __test("test17", set([nn.ATYPE_LIST, nn.ATYPE_FLOATS_LIST]))
    __test("test18", set([nn.ATYPE_LIST, nn.ATYPE_DICTS_LIST]))
    __test("test19", set([nn.ATYPE_STRING]))
    __test("test20", set([nn.ATYPE_LIST, nn.ATYPE_LISTS_LIST]))





  def test_args_required(self):
    """Test raise error on non-existing required arg and success on
    non-required non-existing arg."""
    filename = os.path.join(self.data_path, "valid.json")

    obj = nn.read_json_object(filename)
    val = nn.ArgumentsValidator(obj, "Args Test")

    with self.assertRaises(nn.ArgumentMissingError):
      val.get("non-existing1", nn.ATYPE_NONE, True)

    val.get("non-existing2", nn.ATYPE_NONE, False)






  def test_arg_extra(self):
    """Test raise error on extra args specified."""
    filename = os.path.join(self.data_path, "valid.json")

    obj = nn.read_json_object(filename)
    val = nn.ArgumentsValidator(obj, "Args Test")

    with self.assertRaises(nn.ArgumentUnexpectedError):
      with val:
        val.get("test_string", nn.ATYPE_STRING, True)
        val.get("test_unicode", nn.ATYPE_STRING, True)
        val.get("test_int", nn.ATYPE_INT, True)




  def test_arg_valid(self):
    """Test valid arguments and raise no error."""
    filename = os.path.join(self.data_path, "valid.json")

    obj = nn.read_json_object(filename)
    val = nn.ArgumentsValidator(obj, "Args Test")

    with val:
      val.get("test_string", nn.ATYPE_STRING, True)
      val.get("test_unicode", nn.ATYPE_STRING, True)
      val.get("test_int", nn.ATYPE_INT, True)
      val.get("test_float", nn.ATYPE_FLOAT, True)
      val.get("test_int_list", nn.ATYPE_INTS_LIST, True)
      val.get("test_multi1", [nn.ATYPE_LIST, nn.ATYPE_BOOL], True)
      val.get("test_multi2", [nn.ATYPE_INT, nn.ATYPE_NONE], True)






if __name__ == "__main__":
  unittest.main(verbosity=2)

