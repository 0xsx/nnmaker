# -*- coding: utf-8 -*-
"""
Defines a dictionary of all available models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import nnetmaker.models.convnet0
import nnetmaker.models.pyrnet0
import nnetmaker.models.pyrnet1
import nnetmaker.models.pyrnet2
import nnetmaker.models.pyrnet3
import nnetmaker.models.pyrnet4
import nnetmaker.models.lstm0
import nnetmaker.models.lstm1



MODELS = {
  "ConvNetClassifierModel0": convnet0.ConvNetClassifierModel0,
  "PyramidNetClassifierModel0": pyrnet0.PyramidNetClassifierModel0,
  "PyramidNetClassifierModel1": pyrnet1.PyramidNetClassifierModel1,
  "PyramidNetRenderModel0": pyrnet2.PyramidNetRenderModel0,
  "PyramidNetRegressorModel0": pyrnet3.PyramidNetRegressorModel0,
  "PyramidNetRegressorModel1": pyrnet4.PyramidNetRegressorModel1,
  "LSTMClassifierModel0": lstm0.LSTMClassifierModel0,
  "LSTMClassifierModel1": lstm1.LSTMClassifierModel1
}

