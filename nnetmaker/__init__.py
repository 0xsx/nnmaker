# -*- coding: utf-8 -*-
"""
A Python package for assisting neural network production with TensorFlow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Filter size warnings that might arise from using pre-wheel'd versions of
# packages. These should be safe to ignore.
# See:
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
# https://github.com/numpy/numpy/pull/432
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")




from nnetmaker.batch import *
from nnetmaker.config import *
from nnetmaker.models import MODELS
from nnetmaker.util import *




VERSION = "0.0.1"
INT_VERSION = 0


