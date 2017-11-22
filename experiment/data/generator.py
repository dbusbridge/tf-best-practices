"""Generate the data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def get_data(
    x_min=-4. * np.pi,
    x_max=4. * np.pi,
    n=1024,
    bias=1.,
    noise_sd=0.1,
    y_func=lambda x, b: np.sin(x) + b):
  """Generate x, y data. Defaults to x, y = sin(x) + bias.

  :param float  x_min: Minimum value for x.
  :param float x_max: Maximum value for x.
  :param int n: Number of examples.
  :param float bias: The bias.
  :param int noise_sd: The standard deviation of the normal distribution noise
    model.
  :param y_func: The function to apply to x to return the (noiseless) y values.
  :return: Tuple of x and y values.
  :rtype: tuple(np.array, np.array).
  """
  x = np.linspace(start=x_min, stop=x_max, num=n)
  y = y_func(x, bias)

  noise = np.random.normal(loc=0, scale=noise_sd, size=y.size)

  y += noise

  return x, y
