"""
following: https://www.tensorflow.org/get_started/estimator
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np

import tensorflow as tf


# data sets
IRIS_TRAINING = 'iris_training.csv'
IRIS_TRAINING_URL = 'http://download.tensorflow.org/data/iris_training.csv'

IRIS_TEST = 'iris_test.csv'
IRIS_TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'


def main():
    # get data if needed
    if not os.path.exists(IRIS_TRAINING):
        raw = urllib.urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, 'w') as f:
            f.write(raw)
    if not os.path.exists(IRIS_TEST):
        raw = urllib.urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, 'w') as f:
            f.write(raw)

    # load datasets
    """
    `Dataset`s in `tf.contrib.learn` are _named tuples_. We can access info
    via the `data` and `target` fields - e.g., training_set.data, etc.
    """
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32
    )
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32
    )

    """
    The Estimator API lets us build models from templates quickly. We need
    to provide structure according to the requirements of the API though,
    hence the `tf.feature_column` specification.
    """
    # specify all features have real-valued data
    feature_columns = [tf.feature_column.numeric_column('x', shape=[4])]

    # build a 3-layer DNN with 10, 20, 10 hidden units
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=3,
        model_dir='/tmp/iris_model'
    )

    """
    The Estimator API employs input functions to generate data for the model.
    """
    # define training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True
    )

    """
    note that the state of the model is preserved in the `classifier`, so we
    can train iteratively if we want, e.g.

        classifier.train(input_fn=train_input_fn, steps=1000)
        classifier.train(input_fn=train_input_fn, steps=1000)

    instead of
    
        classifier.train(input_fn=train_input_fn, steps=2000)

    If we wish to track our model while it trains, we should use a TF
    `SessionRunHook` to perform logging ops.
    """
    # train the model
    classifier.train(input_fn=train_input_fn, steps=2000)

    """
    here, use `num_epochs=1` to stop after one loop through the test data.
    """
    # define test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.array(test_set.data)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False
    )

    # evaluate accuracy
    accuracy_score = classifier.evaluate(
        input_fn=test_input_fn
    )['accuracy']

    print('\nTest accuracy: {0:f}\n'.format(accuracy_score))

    # classify two new flower samples
    new_samples = np.array(
        [[6.4, 3.2, 4.5, 1.5],
         [5.8, 3.1, 5.0, 1.7]],
        dtype=np.float32
    )
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': new_samples},
        num_epochs=1,
        shuffle=False
    )
    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p['classes'] for p in predictions]
    print(
        'New samples, class predictions: {}\n'.format(
            predicted_classes
        )
    )


if __name__ == '__main__':
    main()
