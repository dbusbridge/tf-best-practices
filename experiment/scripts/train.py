import os
import shutil

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn import learn_runner
from tensorflow.contrib.training import HParams

import experiment as ex

# Show debugging output
tf.logging.set_verbosity(tf.logging.DEBUG)

# Config
FLAGS = tf.app.flags.FLAGS

# Data parameters
tf.app.flags.DEFINE_string('train_data_dir', '/tmp/output_records/train',
                           'The path containing training TFRecords.')

tf.app.flags.DEFINE_string('eval_data_dir', '/tmp/output_records/valid',
                           'The path containing evaluation TFRecords.')

tf.app.flags.DEFINE_string('model_dir', '/tmp/model/my_first_model',
                           'The path to write the model to.')

tf.app.flags.DEFINE_boolean('clean_model_dir', True,
                            'Whether to start from fresh.')

# Hyperparameters
tf.app.flags.DEFINE_float('learning_rate', 1.e-2,
                          'The learning rate.')

tf.app.flags.DEFINE_integer('batch_size', 64,
                            'The batch size.')

tf.app.flags.DEFINE_integer('epochs', 1000,
                            'Number of epochs to train for.')

tf.app.flags.DEFINE_integer('shuffle', True,
                            'Whether to shuffle dataset.')


# Evaluation
tf.app.flags.DEFINE_integer('min_eval_frequency', 128,
                            'Frequency to do evaluation run.')


# Globals
tf.app.flags.DEFINE_integer('random_seed', 1234,
                            'The extremely random seed.')

tf.app.flags.DEFINE_boolean('use_jit_xla', False,
                            'Whether to use XLA compilation..')

# Hyperparameters
tf.app.flags.DEFINE_string(
  'hyperparameters_path',
  'alignment/models/configurations/single_layer.json',
  'The path to the hyperparameters.')


def run_experiment(unused_argv):
  """Run the training experiment."""
  hyperparameters_dict = FLAGS.__flags

  # Build the hyperparameters object
  params = HParams(**hyperparameters_dict)

  np.random.seed(params.random_seed)
  tf.set_random_seed(params.random_seed)

  # Initialise the run config
  run_config = tf.contrib.learn.RunConfig()

  # Use JIT XLA
  session_config = tf.ConfigProto()
  if params.use_jit_xla:
    session_config.graph_options.optimizer_options.global_jit_level = (
      tf.OptimizerOptions.ON_1)

  # Clean the model directory
  if os.path.exists(params.model_dir) and params.clean_model_dir:
    shutil.rmtree(params.model_dir)

  # Update the run config
  run_config = run_config.replace(tf_random_seed=params.random_seed)
  run_config = run_config.replace(model_dir=params.model_dir)
  run_config = run_config.replace(session_config=session_config)
  run_config = run_config.replace(
    save_checkpoints_steps=params.min_eval_frequency)

  # Output relevant info for inference
  ex.save_dict_json(d=params.values(),
                    path=os.path.join(params.model_dir, 'params.dict'),
                    verbose=True)
  ex.save_obj(obj=params,
              path=os.path.join(params.model_dir, 'params.pkl'), verbose=True)

  learn_runner.run(
    experiment_fn=ex.experiment_fn,
    run_config=run_config,
    schedule='train_and_evaluate',
    hparams=params)


if __name__ == '__main__':
  tf.app.run(main=run_experiment)
