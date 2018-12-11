#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exposes the data path location and the package to be tested, and defines a test
case to verify that the imported package is the correct version and that the
test data path exists.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import unittest

import os
import sys
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

import nnetmaker as nn


TEST_VERSION = "0.0.1"
TEST_INT_VERSION = 0

TEST_DATA_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), "test_data"))





class ImportTestCase(unittest.TestCase):
  def setUp(self):
    self.longMessage = True

  def test_version(self):
    self.assertEqual(TEST_VERSION, nn.VERSION, "incorrect test version string")

  def test_int_version(self):
    self.assertEqual(TEST_INT_VERSION, nn.INT_VERSION, "incorrect test version int")

  def test_data_path_available(self):
    self.assertTrue(os.path.isdir(TEST_DATA_PATH), "test data path unavailable")





if __name__ == "__main__":
  unittest.main(verbosity=2)
else:
  assert TEST_VERSION == nn.VERSION
  assert TEST_INT_VERSION == nn.INT_VERSION
  assert os.path.isdir(TEST_DATA_PATH)
