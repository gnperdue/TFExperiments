"""
Simple logistic regression to test TF framework APIs.

Fashion MNIST TF records available here:
    https://github.com/gnperdue/RandomData/tree/master/TensorFlow

Fashion MNIST HDF5 records available here:
    https://github.com/gnperdue/RandomData/tree/master/hdf5
"""
import time
import os
import argparse
import logging

import tensorflow as tf

from tffashion.data_readers import make_fashion_iterators
from tffashion.tf_model_classes import FashionMNISTLogReg

logging.basicConfig(
    filename='tmp_logreg_log.txt', level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def main(
        data_dir, batch_size, num_epochs, train_steps, model_dir,
        learning_rate
):
    tf.reset_default_graph()

    test_file = os.path.join(data_dir, 'fashion_test.hdf5')
    train_file = os.path.join(data_dir, 'fashion_train.hdf5')
    LOGGER.info('train file = {}, test file = {}'.format(
        train_file, test_file
    ))
    train(train_file, batch_size, num_epochs, train_steps, model_dir,
          learning_rate)
    test(test_file, batch_size, model_dir)


def train(
        train_file, batch_size, num_epochs, train_steps, model_dir,
        learning_rate
):
    tf.reset_default_graph()
    chkpt_dir = model_dir + '/checkpoints'
    run_dest_dir = model_dir + '/%d' % time.time()

    with tf.Graph().as_default() as g:
        with tf.Session(graph=g) as sess:

            features, labels = make_fashion_iterators(
                train_file, batch_size, num_epochs, shuffle=True
            )
            features = tf.reshape(features, [-1, 784])

            model = FashionMNISTLogReg()
            model.build_network(features, labels)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            writer = tf.summary.FileWriter(
                logdir=run_dest_dir, graph=sess.graph
            )
            saver = tf.train.Saver(save_relative_paths=True)

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(chkpt_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                LOGGER.info('Restored session from {}'.format(chkpt_dir))

            writer.add_graph(sess.graph)
            initial_step = model.global_step.eval()
            LOGGER.info('initial step = {}'.format(initial_step))

            for b_num in range(initial_step, initial_step + train_steps):
                _, loss_batch, summary_t = sess.run(
                    [model.optimizer, model.loss, model.train_summary_op]
                )
                LOGGER.info(
                    ' Loss @step {}: {:5.1f}'.format(b_num, loss_batch)
                )
                writer.add_summary(summary_t, global_step=b_num)
                if (b_num + 1) % 5 == 0:
                    saver.save(sess, chkpt_dir, b_num)
                    writer.add_summary(summary_t, global_step=b_num)

        writer.close()


def test(test_file, batch_size, model_dir):
    tf.reset_default_graph()
    chkpt_dir = model_dir + '/checkpoints'

    with tf.Graph().as_default() as g:
        with tf.Session(graph=g) as sess:

            features, labels = make_fashion_iterators(
                test_file, batch_size, num_epochs=1, shuffle=False
            )
            features = tf.reshape(features, [-1, 784])

            model = FashionMNISTLogReg()
            model.build_network(features, labels)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver()

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(chkpt_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                LOGGER.info('Restored session from {}'.format(chkpt_dir))

            average_loss = 0.0
            total_correct_preds = 0
            n_processed = 0

            initial_step = model.global_step.eval()
            LOGGER.info('initial step = {}'.format(initial_step))

            try:
                for i in range(1000000000):
                    loss_batch, logits_batch, Y_batch = sess.run(
                        [model.loss, model.logits, labels]
                    )
                    n_processed += batch_size
                    average_loss += loss_batch
                    preds = tf.nn.softmax(logits_batch)
                    correct_preds = tf.equal(
                        tf.argmax(preds, 1), tf.argmax(Y_batch, 1)
                    )
                    accuracy = tf.reduce_sum(
                        tf.cast(correct_preds, tf.float32)
                    )
                    total_correct_preds += sess.run(accuracy)
                    LOGGER.info('  batch {} loss = {} for nproc {}'.format(
                        i, loss_batch, n_processed
                    ))
                    LOGGER.info("  total_corr_preds / nproc = {} / {}".format(
                        total_correct_preds, n_processed
                    ))
                    LOGGER.info("  Cumul. Accuracy {0}".format(
                        total_correct_preds / n_processed
                    ))
            except tf.errors.OutOfRangeError:
                LOGGER.info('Testing stopped - queue is empty.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir', type=str, required=True,
        help='HDF5 data directory.'
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
        '--train-steps', type=int, default=1e9,
        help='Number of training steps'
    )
    parser.add_argument(
        '--model-dir', type=str, default='chkpts/mnist_graph',
        help='Model directory'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.01,
        help='Learning rate'
    )
    args = parser.parse_args()

    main(**vars(args))
