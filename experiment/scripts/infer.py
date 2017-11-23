import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from experiment.framework.estimator import estimator_fn
from experiment.framework.input import get_input_fn

# Config
FLAGS = tf.app.flags.FLAGS

# Paths
tf.app.flags.DEFINE_string('model_dir', '/tmp/model/my_first_model',
                           'The path the model was written to.')


def infer(unused_argv):
  params_path = os.path.join(FLAGS.model_dir, 'params.pkl')
  with open(params_path, 'rb') as f:
    params = pickle.load(f)

  # Set the seeds
  np.random.seed(params.random_seed)
  tf.set_random_seed(params.random_seed)

  # Set the run_config where to load the model from
  run_config = tf.contrib.learn.RunConfig()
  run_config = run_config.replace(model_dir=FLAGS.model_dir)

  # Initialize the estimator
  estimator = estimator_fn(run_config, params)

  # Get the data directories
  data_dirs = {'training': params.train_data_dir,
               'evaluation': params.eval_data_dir}

  # Get the corresponding input functions
  input_fns = {
    k: get_input_fn(
      data_dir=v, batch_size=params.batch_size, name=k)
    for k, v in data_dirs.items()}

  # Build the prediction generators and extract the predictions
  prediction_generators = {k: estimator.predict(input_fn=v)
                 for k, v in input_fns.items()}

  predictions = {k: list(v) for k, v in prediction_generators.items()}

  # Names of variables
  independent_var = 'x'
  dependent_vars = ['y', 'y_pred']
  variables = [independent_var] + dependent_vars

  # Extract the values for each component. At this point
  variable_values = {
    dataset: {v: np.concatenate(
      [example[v] for example in predictions[dataset]])
      for v in variables}
    for dataset in data_dirs}

  # Build the figure
  fig, ax = plt.subplots(nrows=1, ncols=1)

  for data_set_name, data_set_dict in variable_values.items():
    # Get the values for the independent variable
    independent_var_values = data_set_dict[independent_var]

    for var_name in dependent_vars:
      # Get the values for the specific dependent variable
      dependent_var_values = data_set_dict[var_name]

      # The label for the legend
      plot_label = "{}_{}".format(data_set_name, var_name)

      # Scatter!
      ax.scatter(independent_var_values, dependent_var_values, label=plot_label)

  ax.legend()


if __name__ == "__main__":
  tf.app.run(main=infer)
