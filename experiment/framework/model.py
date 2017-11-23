import tensorflow as tf

from tensorflow.contrib.learn import ModeKeys

from experiment.framework.summary import build_variable_summaries


def model_fn(features, labels, mode, params):
  """Model function used in the estimator.

  :param dict features: Input features dict to the model.
  :param tf.Tensor labels: Labels tensor for training and evaluation.
  :param ModeKeys mode: Specifies if training, evaluation or prediction.
  :param HParams params: The Hyperparameters.
  :return: The model to be run by Estimator.
  :rtype: EstimatorSpec.
  """
  # Are we training?
  training = mode == ModeKeys.TRAIN

  # Apply model architecture to the features.
  logits = _layers(features['x'], training=training)

  # If mode is inference, return the predictions
  if mode == ModeKeys.INFER:
    predictions = {'y_pred': logits}

    # Add features to the predictions for easy later analysis
    predictions.update(features)

    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions)

  # Build the regression head
  head = tf.contrib.estimator.regression_head(name='regression_head')

  # Calculate the loss, training and metric ops
  loss = tf.losses.mean_squared_error(
    labels=labels, predictions=logits)

  # Build the train op
  train_op_fn = _get_train_op_fn(params=params)

  # Build the variable summaries
  build_variable_summaries(gradients=True, loss=loss)

  return head.create_estimator_spec(
    features=features, mode=mode,
    logits=logits, labels=labels,
    train_op_fn=train_op_fn)


def _layers(inputs, training, name='mlp'):
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
      inputs=inputs, units=100, activation=tf.nn.relu, name='dense_1')
    # net = tf.layers.dropout(inputs=net, training=training, name='dropout_1')
    net = tf.layers.dense(
      inputs=net, units=100, activation=tf.nn.relu, name='dense_2')
    # net = tf.layers.dropout(inputs=net, training=training, name='dropout_2')
    net = tf.layers.dense(inputs=net, units=1, name='dense_3')

  return net


def _get_train_op_fn(params):
  """Get the training op function.

  :param HParams params: Hyperparameters (needs to have `learning_rate`).
  :return: The training op function.
  """
  def train_op_fn(loss):
    """Get the training op.

    :param tf.Tensor loss: Scalar Tensor that represents the loss function.
    :return: The training op.
    """
    return tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.train.get_global_step(),
      optimizer=tf.train.AdamOptimizer,
      learning_rate=params.learning_rate)

  return train_op_fn


def _get_eval_metric_ops(labels, predictions):
  """Get a dict of the evaluation Ops.

  :param tf.Tensor labels: Labels tensor for training and evaluation.
  :param tf.Tensor predictions: Predictions Tensor.
  :return: Dict of metric results keyed by name.
  :rtype: dict.
   """
  return {
    'Mean squared error': tf.metrics.mean_squared_error(
      labels=labels,
      predictions=predictions,
      name='mse')}
