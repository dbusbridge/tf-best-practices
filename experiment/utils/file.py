import json
import os
import pickle

import tensorflow as tf


def create_not_exist(path, verbose=False):
  if not os.path.exists(path):
    if verbose:
      tf.logging.info("Creating {p}".format(p=path))
    os.makedirs(path)


def save_obj(obj, path, verbose=False):
  if verbose:
    tf.logging.info("Saving object to {p}".format(p=path))
  with open(path, 'wb') as handle:
    pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_dict_as_json(d, path, verbose=False):
  if verbose:
    tf.logging.info("Saving dict as json to {p}".format(p=path))
  with open(path, 'w') as handle:
    json.dump(d, handle)


def save_dict_json(d, path, verbose=False):
  # Create folder if it does not exist
  create_not_exist(path=os.path.dirname(path), verbose=verbose)
  save_obj(d, '{p}.pkl'.format(p=path), verbose=verbose)
  save_dict_as_json(d, '{p}.json'.format(p=path), verbose=verbose)
