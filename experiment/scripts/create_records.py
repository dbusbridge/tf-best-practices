"""Converts sin data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

import experiment as ex


# Config
FLAGS = tf.app.flags.FLAGS


# Data parameters
tf.app.flags.DEFINE_string('output_dir', '/tmp/output_records',
                           'The path to output the records to.')

tf.app.flags.DEFINE_float('validation_fraction', 0.33,
                          'The fraction of the data set to use for validation.')

tf.app.flags.DEFINE_integer('train_shards', 3,
                            'Number of shards for the training set.')

tf.app.flags.DEFINE_integer('valid_shards', 2,
                            'Number of shards for the validation set.')

# Globals
tf.app.flags.DEFINE_integer('random_seed', 1234,
                            'The extremely random seed.')

# Set the random seeds
tf.set_random_seed(FLAGS.random_seed)
np.random.seed(FLAGS.random_seed)

# Logging verbosity to INFO
tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
  # Get the data.
  x, y = ex.get_data()

  # Split into training and validation.
  x_train, x_valid, y_train, y_valid = train_test_split(
    x, y, test_size=FLAGS.validation_fraction)

  data_sets = {'train': zip(x_train, y_train), 'valid': zip(x_valid, y_valid)}

  shards = {'train': FLAGS.train_shards, 'valid': FLAGS.valid_shards}

  data_sets_serialised = {name: [ex.get_serialized_example(x, y)
                                 for x, y in dataset]
                          for name, dataset in data_sets.items()}

  # Write out to TFRecords.
  for name, dataset in data_sets_serialised.items():
    ex.write_dataset(output_dir=FLAGS.output_dir,
                     name=name, dataset=dataset, num_shards=shards[name])

  tf.logging.info("Finished writing datasets.")


if __name__ == '__main__':
  tf.app.run(main=main)
