"""
Autoencoder using the TF graph APIs
"""
import time
import os
import argparse
import logging

import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tffashion.data_readers import make_fashion_iterators
from tffashion.tf_model_classes import FashionAutoencoder

logging.basicConfig(
    filename='tmp_logreg_log.txt', level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


# TODO - move HDF5 manip functions to a module
# TODO - encode to binary
def setup_hdf5(file_name, n_encoded, n_labels, encoded_dtype='float32'):
    if os.path.exists(file_name):
        os.remove(file_name)

    f = h5py.File(file_name, 'w')
    grp = f.create_group('encoded')
    grp.create_dataset(
        'images', (0, n_encoded), dtype=encoded_dtype, compression='gzip',
        maxshape=(None, n_encoded)
    )
    grp.create_dataset(
        'labels', (0, n_labels), dtype='uint8', compression='gzip',
        maxshape=(None, n_labels)
    )
    return f


def add_batch_to_hdf5(f, encoded_set, labels_set):
    assert len(encoded_set) == len(labels_set), "data length mismatch"
    existing_examples = np.shape(f['encoded/images'])[0]
    total_examples = len(encoded_set) + existing_examples
    f['encoded/images'].resize(total_examples, axis=0)
    f['encoded/labels'].resize(total_examples, axis=0)
    f['encoded/images'][existing_examples: total_examples] = encoded_set
    f['encoded/labels'][existing_examples: total_examples] = labels_set
    return total_examples


def main(
        data_dir, batch_size, num_epochs, train_steps, model_dir,
        learning_rate
):
    tf.reset_default_graph()

    test_file = os.path.join(data_dir, 'fashion_test.hdf5')
    train_file = os.path.join(data_dir, 'fashion_train.hdf5')
    print(train_file, test_file)
    train(train_file, batch_size, num_epochs, train_steps, model_dir,
          learning_rate)
    n_encoded, n_labels = test(test_file, model_dir)
    encode(test_file, model_dir, 'encoded_test.hdf5', n_encoded, n_labels)


def train(
        train_file, batch_size, num_epochs, train_steps, model_dir,
        learning_rate
):
    tf.reset_default_graph()
    chkpt_dir = model_dir + '/checkpoints'
    run_dest_dir = model_dir + '/%d' % time.time()
    n_steps = train_steps or 1000000000

    with tf.Graph().as_default() as g:
        with tf.Session(graph=g) as sess:

            features, _ = make_fashion_iterators(
                train_file, batch_size, num_epochs, shuffle=True
            )
            model = FashionAutoencoder(learning_rate=learning_rate)
            model.build_network(features)

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

            try:
                for b_num in range(initial_step, initial_step + n_steps):
                    _, loss_batch, encoded, summary_t = sess.run(
                        [model.optimizer,
                         model.loss,
                         model.encoded,
                         model.train_summary_op]
                    )
                    if (b_num + 1) % 50 == 0:
                        LOGGER.info(
                            ' Loss @step {}: {:5.1f}'.format(b_num, loss_batch)
                        )
                        LOGGER.debug(str(encoded))
                        saver.save(sess, chkpt_dir, b_num)
                        writer.add_summary(summary_t, global_step=b_num)

            except tf.errors.OutOfRangeError:
                LOGGER.info('Training stopped - queue is empty.')

            saver.save(sess, chkpt_dir, b_num)
            writer.add_summary(summary_t, global_step=b_num)

        writer.close()


def test(test_file, model_dir):
    tf.reset_default_graph()
    chkpt_dir = model_dir + '/checkpoints'

    with tf.Graph().as_default() as g:
        with tf.Session(graph=g) as sess:

            features, labels = make_fashion_iterators(
                test_file, batch_size=1, num_epochs=1, shuffle=False
            )
            model = FashionAutoencoder()
            model.build_network(features)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver()

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(chkpt_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                LOGGER.info('Restored session from {}'.format(chkpt_dir))

            initial_step = model.global_step.eval()
            LOGGER.info('initial step = {}'.format(initial_step))

            try:
                for i in range(20):
                    loss_batch, encoded_batch, labels_batch, input, recon = \
                        sess.run(
                            [model.loss,
                             model.encoded,
                             labels,
                             model.X,
                             model.Y]
                        )
                    print(loss_batch, encoded_batch.shape, recon.shape)
                    n_encoded = encoded_batch.shape[1]
                    n_labels = labels_batch.shape[1]

                    fig = plt.figure()
                    gs = plt.GridSpec(1, 3)
                    ax1 = plt.subplot(gs[0])
                    ax1.imshow(recon[0].reshape(28, 28))
                    ax2 = plt.subplot(gs[1])
                    ax2.imshow(input[0].reshape(28, 28))
                    plt.title(np.argmax(labels_batch[0]))
                    ax3 = plt.subplot(gs[2])
                    ax3.imshow(encoded_batch[0].reshape(8, 8))
                    figname = 'image_{:04d}.pdf'.format(i)
                    plt.savefig(figname, bbox_inches='tight')
                    plt.close()

            except tf.errors.OutOfRangeError:
                LOGGER.info('Testing stopped - queue is empty.')

    return n_encoded, n_labels


def encode(data_file, model_dir, encoded_file_name, n_encoded, n_labels):
    tf.reset_default_graph()
    chkpt_dir = model_dir + '/checkpoints'

    with tf.Graph().as_default() as g:
        with tf.Session(graph=g) as sess:

            features, labels = make_fashion_iterators(
                data_file, batch_size=50, num_epochs=1, shuffle=False
            )
            model = FashionAutoencoder()
            model.build_network(features)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver()

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(chkpt_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                LOGGER.info('Restored session from {}'.format(chkpt_dir))

            initial_step = model.global_step.eval()
            LOGGER.info('initial step = {}'.format(initial_step))

            f = setup_hdf5(encoded_file_name, n_encoded, n_labels)

            try:
                for i in range(1000000000):
                    encoded_batch, labels_batch = sess.run([
                        model.encoded, labels
                    ])
                    add_batch_to_hdf5(f, encoded_batch, labels_batch)
            except tf.errors.OutOfRangeError:
                LOGGER.info('Testing stopped - queue is empty.')

            f.close()


def decode():
    # don't know if we _really_ need this
    pass


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
        '--train-steps', type=int, default=None,
        help='Number of training steps'
    )
    parser.add_argument(
        '--model-dir', type=str, default='chkpts/mnist_graph',
        help='Model directory'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.0001,
        help='Learning rate'
    )
    args = parser.parse_args()

    main(**vars(args))
