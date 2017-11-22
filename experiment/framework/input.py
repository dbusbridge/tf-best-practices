import os

import tensorflow as tf


# Use `tf.parse_single_example()` to extract data from a `tf.Example`
# protocol buffer, and perform any additional per-record preprocessing.
def _parser(record):
  """Parser for TFRecord."""
  keys_to_features = {
    "x": tf.FixedLenFeature((), tf.float32,
                            default_value=tf.zeros([], dtype=tf.float32)),
    "y": tf.FixedLenFeature((), tf.float32,
                            default_value=tf.zeros([], dtype=tf.float32))}
  return tf.parse_single_example(record, keys_to_features)


def get_input_fn(data_dir, batch_size, epochs, shuffle=True, name="data"):
  """Build the input function from the .tfrecords files in a directory.

  :param str data_dir: The directory containing the TFRecords files. These must
    have the file extension `.tfrecords`.
  :param int batch_size: The batch size.
  :param int epochs: The number of epochs.
  :param bool shuffle: Whether to shuffle the data set. Defaults to `True`.
  :param str name: The name of the data set for variable name scoping. Defaults
    to 'data'.
  :return: The tensors corresponding to the values for x and y provided by the
    generator.
  :rtype: tuple(tf.Tensor, tf.Tensor).
  """
  def input_fn():
    with tf.name_scope(name):
      filenames = [os.path.join(data_dir, x)
                   for x in os.listdir(path=data_dir)
                   if x.endswith(".tfrecords")]
      dataset = tf.data.TFRecordDataset(filenames)

      # Use `Dataset.map()` to build a pair of a feature dictionary and a label
      # tensor for each example.
      dataset = dataset.map(_parser)

      if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
      dataset = dataset.batch(batch_size)
      dataset = dataset.repeat(epochs)
      iterator = dataset.make_one_shot_iterator()
      next_examples = iterator.get_next()

      x, y = next_examples['x'], next_examples['y']

      # Make them batches
      x, y = tf.expand_dims(x, axis=-1), tf.expand_dims(y, axis=-1)

      return x, y
  return input_fn
