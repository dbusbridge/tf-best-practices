# Best practices in TensorFlow (how to use the Estimator and Experiments API)

## What

[TensorFlow](https://www.tensorflow.org/) allows (relatively) easy creation and
optimisation of computational graphs for a practically unlimited range of 
tasks. For each of these tasks, however, one needs:

+ An input pipeline
+ A model specification
+ A training loop
+ An evaluation loop
+ Metrics
+ ...hundreds of other things but I think the point is clear.

To avoid the substantial engineering challenge of implementing all of these 
components each time you want to train a cool new model, TensorFlow now includes
the 
[Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator) 

![estimator](images/estimator.png)

(Image taken from https://arxiv.org/abs/1708.02637)

and [Experiments](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment)
API. On top of this, to help with the input pipeline, the 
[Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) has also 
now exists.

This repository aims to provide a clear template on how to use each of these 
ideas in tandem to create a clean, efficient training and evaluation setup. 

## Code

The code contains the following scripts to be run in order:

+ `experiment/scripts/create_records.py` Creates `x, y` examples of a biased sin 
function with noise. It saves these in a `data_dir`  as sharded `.tfrecords` 
files. Folder for training and validation shards to ensure data separation from 
the start.
+ `experiment/scripts/train.py` Launches an `Experiment`, initialising the 
training and validation loop, building a distinct `Estimator` instance for each 
loop. Data is read from the `data_dir`. The model is saved in a `model_dir`.
+ `experiment/scripts/infer.py` Reloads the `Estimator` from the `model_dir` and
creates a plot of the predictions on the training and validation set.

![experiment](images/experiment.png)

Image taken from https://arxiv.org/abs/1708.02637


## References

### Blog posts

+ Great introductory post by Peter Roelants [Higher-Level APIs in TensorFlow](https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0).

### Papers

+ Cheng, H.-T. et al. (2017). 
TensorFlow Estimators: Managing Simplicity vs. Flexibility in High-Level Machine Learning Frameworks. 
  + DOI: https://doi.org/10.1145/3097983.3098171 
  + arXiv: http://arxiv.org/abs/1708.02637
