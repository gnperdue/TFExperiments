from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import tensorflow as tf
from tfconv.data_readers import make_mnist_iterators
from tfconv.estimator_funcs import mnist_conv

tf.logging.set_verbosity(tf.logging.DEBUG)


def main(hdf5_path, batch_size, num_epochs, train_steps):
    import os
    data = {}
    for dset in ['test', 'train', 'valid']:
        data[dset] = os.path.join(
            hdf5_path, 'mnist_{}.hdf5'.format(dset)
        )
        print('data[{}] = {}'.format(dset, data[dset]))

    # checkpoint config - can't have both steps and secs, sadly
    # if model_dir is set, it must agree with estimator if set there also
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=10,
        keep_checkpoint_max=3,
        model_dir='models/mnist'
    )

    classifier = tf.estimator.Estimator(
        model_fn=mnist_conv,
        params={
            'use_dropout': True,
            'dropout_keep_prob': 0.75,
            'learning_rate': 0.001
        },
        config=run_config,
    )
    
    classifier.train(
        input_fn=lambda: make_mnist_iterators(
            data['train'], batch_size, num_epochs
        ),
        steps=train_steps
    )

    eval_result = classifier.evaluate(
        input_fn=lambda: make_mnist_iterators(
            data['valid'], batch_size, num_epochs
        ),
        steps=100,
    )
    print('\nEval set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hdf5-path', type=str, required=True,
        help='Path to the MNIST HDF5 data.'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--num-epochs', type=int, default=1,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--train-steps', type=int, default=None,
        help='Number of training steps'
    )
    args = parser.parse_args()

    main(**vars(args))
