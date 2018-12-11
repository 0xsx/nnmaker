# -*- coding: utf-8 -*-
"""
Defines a deep bi-directional LSTM neural network with peephole
connections for classification for continuous datasets, using the
prediction of the final timestep as target.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np
import tensorflow as tf


from nnetmaker.model import *
from nnetmaker.util import *


from nnetmaker.models.lstm0 import LSTMClassifierModel0



class LSTMClassifierModel1(LSTMClassifierModel0):





  def _build_cost_targets(self, in_vars, target_vars, out_vars, **kwargs):

    cost_targets = []
    cost_targets.append((target_vars["predictions"][:, -1, :], out_vars["predictions"], None))

    return cost_targets






  def _build_metrics(self, in_vars, target_vars, out_vars, **kwargs):
    metrics = {}

    targets = tf.argmax(target_vars["predictions"][:, -1, :], axis=1)
    predicted = tf.argmax(out_vars["predictions"], axis=1)

    metrics["accuracy"] = tf.metrics.accuracy(targets, predicted)
    
    return metrics






