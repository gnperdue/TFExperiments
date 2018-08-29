from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf

# from tffashion.estimator_fns import make_shallow_keras_estimator
from tffashion.data_readers import make_fashion_iterators
from tffashion.estimator_fns import shallow_model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=None, type=int,
                    help='number of training steps')
parser.add_argument('--n_epochs', default=1, type=int, help='number of epochs')

# Get path to data
TESTFILE = os.path.join(
    os.environ['HOME'], 'Dropbox/Data/RandomData/hdf5/fashion_test.hdf5'
)
TRAINFILE = os.path.join(
    os.environ['HOME'], 'Dropbox/Data/RandomData/hdf5/fashion_train.hdf5'
)


def predict(classifier, batch_size):
    # predictions is a generator - evaluation is lazy
    predictions = classifier.predict(
        input_fn=lambda: make_fashion_iterators(
            TESTFILE, batch_size, num_epochs=1
        ),
    )
    counter = 0
    for p in predictions:
        print(p)
        counter += 1
        if counter > 10:
            break


def evaluate(classifier, batch_size):
    eval_result = classifier.evaluate(
        input_fn=lambda: make_fashion_iterators(
            TESTFILE, batch_size, num_epochs=1
        ),
        steps=100,
    )
    print('\nEval:')
    print('acc: {accuracy:0.3f}, loss: {loss:0.3f}, MPCA {mpca:0.3f}'.format(
        **eval_result
    ))


def train_one_epoch(classifier, batch_size, train_steps):
    classifier.train(
        input_fn=lambda: make_fashion_iterators(
            TRAINFILE, batch_size, num_epochs=1
        ),
        steps=train_steps
    )


def train(classifier, batch_size, num_epochs, train_steps):
    for i in range(num_epochs):
        print('training epoch {}'.format(i))
        train_one_epoch(classifier, batch_size, train_steps)
        print('evaluation after epoch {}'.format(i))
        evaluate(classifier, batch_size)


def main(batch_size, train_steps, n_epochs):

    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=10,
        save_summary_steps=10,
        keep_checkpoint_max=3,
        model_dir='/tmp/fashion',
        tf_random_seed=None,
    )
    classifier = tf.estimator.Estimator(
        model_fn=shallow_model_fn,
        params={},
        config=run_config
    )
    train(classifier, batch_size, n_epochs, train_steps)
    predict(classifier, batch_size)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
