import tensorflow as tf

from tensorflow.contrib.learn import ModeKeys

from experiment.framework.summary import build_variable_summaries


def estimator_fn(run_config, params):
  """Return the model as a Tensorflow Estimator object.

  :param RunConfig run_config: Configuration for Estimator run.
  :param HParams params: The hyperparameters.
  :return: The Estimator,
  :rtype: Estimator.
  """
  return tf.estimator.Estimator(
    model_fn=_model_fn,
    params=params,
    config=run_config)


def _architecture(inputs, training, name='mlp'):
  """The model architecture.

  :param tf.Tensor inputs: The inputs
  :param bool training: `True` if training, `False` otherwise. Used to control
    dropout layers.
  :param str name: Name of the variable scope.
  :return: The outputs..
  :rtype: tf.Tensor.
  """
  with tf.variable_scope(name):
    net = tf.layers.dense(
      inputs=inputs, units=10, activation=tf.nn.relu, name='dense_1')
    net = tf.layers.dropout(inputs=net, training=training, name='dropout_1')
    net = tf.layers.dense(inputs=net, units=1, name='dense_2')

  return net


def _model_fn(features, labels, mode, params):
  """Model function used in the estimator.

  :param tf.Tensor features: Input features to the model.
  :param tf.Tensor labels: Labels tensor for training and evaluation.
  :param ModeKeys mode: Specifies if training, evaluation or prediction.
  :param HParams params: The Hyperparameters.
  :return: The model to be run by Estimator.
  :rtype: EstimatorSpec.
  """
  # Are we training?
  training = mode == ModeKeys.TRAIN

  # Apply model architecture to the features.
  predictions = _architecture(features, training=training)

  # If mode is inference, return the predictions
  if mode == ModeKeys.INFER:
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions)

  # Calculate the loss, training and metric ops
  loss = tf.losses.mean_squared_error(
    labels=labels,
    predictions=predictions)
  train_op = _train_op_fn(loss, params)
  eval_metric_ops = _eval_metric_ops(labels, predictions)

  # Build the variable summaries
  build_variable_summaries(gradients=True, loss=loss)

  return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=predictions,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=eval_metric_ops)


def _train_op_fn(loss, params):
  """Get the training op.

  :param tf.Tensor loss: Scalar Tensor that represents the loss function.
  :param HParams params: Hyperparameters (needs to have `learning_rate`).
  :return: The training op.
  """
  return tf.contrib.layers.optimize_loss(
    loss=loss,
    global_step=tf.train.get_global_step(),
    optimizer=tf.train.AdamOptimizer,
    learning_rate=params.learning_rate)


def _eval_metric_ops(labels, predictions):
  """Return a dict of the evaluation Ops.
  Args:
    labels (Tensor): Labels tensor for training and evaluation.
    predictions (Tensor): Predictions Tensor.
  Returns:
    Dict of metric results keyed by name.
  """
  return {
    'Mean squared error': tf.metrics.mean_squared_error(
      labels=labels,
      predictions=predictions,
      name='mse')}
