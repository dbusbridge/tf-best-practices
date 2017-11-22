import tensorflow as tf

from experiment.framework.estimator import estimator_fn
from experiment.framework.input import get_input_fn


def experiment_fn(run_config, params):
  """Create an experiment to train and evaluate the model.
  Args:
    run_config (RunConfig): Configuration for Estimator run.
    params (HParam): Hyperparameters
  Returns:
    (Experiment) Experiment for training the mnist model.
  """
  # Get the estimator
  estimator = estimator_fn(run_config=run_config, params=params)

  # Get the inputs
  train_input_fn = get_input_fn(
    data_dir=params.train_data_dir,
    batch_size=params.batch_size, epochs=params.epochs, shuffle=params.shuffle,
    name='training_data')

  eval_input_fn = get_input_fn(
    data_dir=params.eval_data_dir,
    batch_size=params.batch_size, epochs=params.epochs, shuffle=params.shuffle,
    name='evaluation_data')

  # Create the experiment
  return tf.contrib.learn.Experiment(
    estimator=estimator,
    train_input_fn=train_input_fn,
    eval_input_fn=eval_input_fn,
    min_eval_frequency=params.min_eval_frequency)
