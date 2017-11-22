"""Write the data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf


def write_dataset(output_dir, name, dataset, num_shards):
  """Write a sharded TFRecord dataset.

  :param str output_dir: Name of the data output directory.
  :param str name: Name of the dataset (e.g. "train").
  :param list dataset: List of serialized Example protos for that dataset.
  :param int num_shards: The number of output shards for that dataset.
  """
  tf.logging.info("Writing dataset {}.".format(name))

  name_folder = os.path.join(output_dir, name)
  os.makedirs(name_folder, exist_ok=True)

  shard_indices_all = np.array_split(np.arange(len(dataset)), num_shards)

  for i, shard_indices in enumerate(shard_indices_all):
    filename = os.path.join(
      output_dir, name,
      "{}-of-{}.tfrecords".format(i + 1, num_shards))

    _write_shard(filename, dataset, shard_indices)

    tf.logging.info("Wrote dataset shard {}.".format(filename))

  tf.logging.info("Finished writing shards for dataset {}.".format(name))


def _write_shard(filepath, dataset, indices):
  """Write a TFRecord shard.

  :param str filepath: The path to write the shard to.
  :param list dataset: List of serialized Example protos for that dataset.
  :param iterable indices: Indices of dataset to write to this shard.
  """
  with tf.python_io.TFRecordWriter(filepath) as writer:
    for j in indices:
      writer.write(dataset[j])
