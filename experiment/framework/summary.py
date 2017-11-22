import tensorflow as tf


def _formatted_varname(var_type):
  return "{}".format(var_type).replace(':', '_').replace(' ', '_')


def _variable_summary(var):
  """Attach a lot of summaries to a Tensor
  (for TensorBoard visualization)."""
  with tf.variable_scope(_formatted_varname(var.name)):
    mean_v = tf.reduce_mean(var, name='mean')
    min_v = tf.reduce_min(var, name='min')
    max_v = tf.reduce_max(var, name='max')
    stddev_v = tf.sqrt(tf.reduce_mean(tf.square(var - mean_v)), name='stddev')

    tf.summary.scalar(_formatted_varname('mean'), mean_v)
    tf.summary.scalar(_formatted_varname('min'), min_v)
    tf.summary.scalar(_formatted_varname('max'), max_v)
    tf.summary.scalar(_formatted_varname('stddev'), stddev_v)

    if "embed" not in var.name:
      if len(var.shape) > 1:
        tf.summary.histogram(_formatted_varname('hist'), var)


def build_variable_summaries(gradients=False, loss=None):
  trainable_variables = tf.trainable_variables()
  for var in trainable_variables:
    _variable_summary(var)

  if gradients:
    assert loss is not None, (
      "If gradient summaries are required, you must specify a loss.")

  if gradients:
    gradients = tf.gradients(loss, trainable_variables)
    for grad, var in list(zip(gradients, trainable_variables)):
      _variable_summary(grad)
