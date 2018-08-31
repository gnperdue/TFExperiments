from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf

from tffashion.model_classes import ConvFashionModel
from tffashion.data_readers import make_fashion_dset

tfe = tf.contrib.eager
tf.logging.set_verbosity(tf.logging.DEBUG)


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=100, type=int, help='batch size')
parser.add_argument('--num-epochs', default=1, type=int,
                    help='number of training epochs')
parser.add_argument('--data-dir', default='', type=str, help='data dir')
parser.add_argument('--model-dir', default='fashion', type=str,
                    help='model dir')


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
    return loss_value, tape.gradient(loss_value, model.variables)


def train(
    model, optimizer, dataset, global_step, checkpoint, checkpoint_prefix
):
    # must include a record_summaries_method
    with tf.contrib.summary.record_summaries_every_n_global_steps(20):
        for (i, (images, labels)) in enumerate(dataset):
            global_step.assign_add(1)
            train_loss, grads = grad(model, images, labels)
            optimizer.apply_gradients(
                zip(grads, model.variables), global_step=global_step
            )
            tf.contrib.summary.scalar('loss', train_loss)
            if i % 20 == 0:
                print('loss at step {:03d}: {:.3f}'.format(i, train_loss))
                checkpoint.save(file_prefix=checkpoint_prefix)
        checkpoint.save(file_prefix=checkpoint_prefix)


def test(model, dataset):
    avg_loss = tfe.metrics.Mean('loss', dtype=tf.float32)
    accuracy = tfe.metrics.Accuracy('accuracy', dtype=tf.float32)

    for (images, labels) in dataset:
        logits = model(images)
        avg_loss(loss(model, images, labels))
        accuracy(
            tf.argmax(logits, axis=1, output_type=tf.int32),
            tf.argmax(labels, axis=1, output_type=tf.int32)
        )

    print('Test set: Average loss: %.4f, Accuracy: %4f%%\n' %
          (avg_loss.result(), 100 * accuracy.result()))
    # need a separate writer (either `with`or as default) to keep distinct
    # with tf.contrib.summary.always_record_summaries():
    #     tf.contrib.summary.scalar('loss', avg_loss.result())
    #     tf.contrib.summary.scalar('accuracy', accuracy.result())


def main(batch_size, num_epochs, data_dir, model_dir):
    tf.enable_eager_execution()

    # Get path to data
    TESTFILE = os.path.join(data_dir, 'fashion_test.hdf5')
    TRAINFILE = os.path.join(data_dir, 'fashion_train.hdf5')

    model = ConvFashionModel()
    dataset = make_fashion_dset(
        TRAINFILE, batch_size, num_epochs=num_epochs, shuffle=True
    )

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

    # writer _can_ make its own log directory
    writer = tf.contrib.summary.create_file_writer(model_dir)
    global_step = tf.train.get_or_create_global_step()
    # writer.set_as_default()  # use a scope instead...

    # os.makedirs(model_dir)   # model dir must exist
    checkpoint_prefix = os.path.join(model_dir, 'ckpt')
    checkpoint = tfe.Checkpoint(
        optimizer=optimizer, model=model, optimizer_step=global_step
    )
    checkpoint.restore(tf.train.latest_checkpoint(model_dir))

    x, y = iter(dataset).next()
    print('initial loss: {:.3f}'.format(loss(model, x, y)))

    # training loop
    with writer.as_default():
        train(
            model, optimizer, dataset, global_step,
            checkpoint, checkpoint_prefix
        )

    test_dataset = make_fashion_dset(TESTFILE, batch_size, num_epochs=1)
    test(model, test_dataset)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
