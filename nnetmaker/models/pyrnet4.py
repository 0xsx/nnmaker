# -*- coding: utf-8 -*-
"""
Defines a deep pyramidal bottleneck residual convolutional neural network for
continuous datasets, using the prediction of the final timestep as target.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np
import tensorflow as tf

from nnetmaker.model import *
from nnetmaker.util import *

from nnetmaker.models.pyrnet3 import PyramidNetRegressorModel0


class PyramidNetRegressorModel1(PyramidNetRegressorModel0):


  def _build_cost_targets(self, in_vars, target_vars, out_vars, **kwargs):

    cost_targets = []
    cost_targets.append((target_vars["predictions"][:, -1], out_vars["predictions"], None))

    return cost_targets






