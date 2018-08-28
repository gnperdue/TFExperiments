from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf

from tffashion.model_classes import ShallowFashionModel
from tffashion.data_readers import make_fashion_iterators

tfe = tf.contrib.eager
tf.logging.set_verbosity(tf.logging.DEBUG)


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

# Get path to data
TESTFILE = os.path.join(
    os.environ['HOME'], 'Dropbox/Data/RandomData/hdf5/fashion_test.hdf5'
)
TRAINFILE = os.path.join(
    os.environ['HOME'], 'Dropbox/Data/RandomData/hdf5/fashion_train.hdf5'
)


def loss(model, x, y):
    prediction = model(x)
    return tf.losses.softmax_cross_entropy(
        onehot_labels=y, logits=prediction
    )
    # return tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits_v2(
    #         logits=prediction, labels=y
    #     )
    # )


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


def main(batch_size, train_steps):
    tf.enable_eager_execution()
    model = ShallowFashionModel()
    train_img, train_lab = make_fashion_iterators(
        TRAINFILE, batch_size, num_epochs=1
    )

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

    print('initial loss: {:.3f}'.format(loss(model, train_img, train_lab)))


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
