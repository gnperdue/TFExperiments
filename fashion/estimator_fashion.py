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


def main(batch_size, train_steps, n_epochs):

    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=10,
        keep_checkpoint_max=3,
        model_dir='/tmp/estimator_fashion'
    )
    classifier = tf.estimator.Estimator(
        model_fn=shallow_model_fn,
        params={},
        config=run_config
    )

    classifier.train(
        input_fn=lambda: make_fashion_iterators(
            TRAINFILE, batch_size, num_epochs=n_epochs
        ),
        steps=train_steps
    )
    eval_result = classifier.evaluate(
        input_fn=lambda: make_fashion_iterators(
            TESTFILE, batch_size, num_epochs=1
        ),
        steps=100,
    )
    print('\nEval set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
