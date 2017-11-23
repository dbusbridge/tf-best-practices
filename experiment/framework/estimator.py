import tensorflow as tf

from experiment.framework.model import model_fn


def estimator_fn(run_config, params):
  """Return the model as a Tensorflow Estimator object.

  :param RunConfig run_config: Configuration for Estimator run.
  :param HParams params: The hyperparameters.
  :return: The Estimator,
  :rtype: Estimator.
  """
  return tf.estimator.Estimator(
    model_fn=model_fn,
    params=params,
    config=run_config)
