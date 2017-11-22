"""Helpers for feature creation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _float_feature(value):
  """Helper for creating a FloatList Feature."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def get_serialized_example(x, y):
  """Helper for creating a serialized Example proto."""
  example = tf.train.Example(
    features=tf.train.Features(
      feature={
        'x': _float_feature(x),
        'y': _float_feature(y)}))

  return example.SerializeToString()
