"""
Logistic regression to test TF framework APIs.

MNIST TF records available here:
    https://github.com/gnperdue/RandomData/tree/master/TensorFlow

MNISFT HDF5 records available here:
    /Users/perdue/Dropbox/Data/RandomData/hdf5
"""
import tensorflow as tf

import time
import os
import logging

from tfmnist.models import MNISTLogReg as MNISTModel
# from tfmnist.data_readers import make_mnist_hdf5iterators as batch_generator
from tfmnist.data_readers import make_mnist_tfreciterators as batch_generator
from tfmnist.summaries import create_or_add_summaries_op

tf.logging.set_verbosity(tf.logging.DEBUG)

logging.basicConfig(
    filename='tmp_logreg_log.txt', level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def create_full_summary_op(scope_name, loss_op, accuracy_op):
    with tf.variable_scope(scope_name, reuse=True):
        summary_op = create_or_add_summaries_op(
            scope_name, 'loss', loss_op
        )
        summary_op = create_or_add_summaries_op(
            scope_name, 'accuracy', accuracy_op
        )
    return summary_op
    

def get_ckpt_and_run_dest(model_dir):
    ckpt_dir = model_dir + '/checkpoints'
    run_dest_dir = model_dir + '/%d' % time.time()
    return ckpt_dir, run_dest_dir


def get_loss_and_acc_ops(model, features, targets):
    logits = model.logits(features)
    sftmx_predictions = model.softmax_predictions(logits)
    loss_op = model.loss(logits, targets)
    accuracy_op = model.accuracy(sftmx_predictions, targets)
    return loss_op, accuracy_op


def validate_one_epoch(
        validation_file, ckpt_dir, run_dest_dir
):
    n_batches = 10
    tf.reset_default_graph()

    with tf.Graph().as_default() as g:
        with tf.Session(graph=g) as sess:

            # set up model
            model = MNISTModel()
            with tf.variable_scope('input'):
                features, targets = batch_generator(
                    validation_file, batch_size=128, num_epochs=1,
                    use_oned_data=True
                )

            # define all ops
            loss_op, accuracy_op = get_loss_and_acc_ops(
                model, features, targets
            )
            gstep_tensr = tf.train.get_or_create_global_step()
            summary_op = create_full_summary_op(
                'summaries/valid', loss_op, accuracy_op
            )

            # set up writer _after_ you've set up vars to be saved
            writer = tf.summary.FileWriter(run_dest_dir, graph=g)
            saver = tf.train.Saver(save_relative_paths=True)

            # variable initialization
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # checkpoint and state management
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                LOGGER.info('Restored session from {}'.format(ckpt_dir))

            initial_step = gstep_tensr.eval()
            LOGGER.info('initial step = {}'.format(initial_step))

            # validate
            try:
                for b_num in range(initial_step, initial_step + n_batches):
                    loss_batch, summary_t = sess.run([loss_op, summary_op])
                    LOGGER.info(
                        'Validation Loss @step {}: {:5.1f}'.format(
                            b_num, loss_batch
                        )
                    )
                    writer.add_summary(summary_t, global_step=b_num)
            except tf.errors.OutOfRangeError:
                LOGGER.info('Validation stopped - queue is empty.')

        writer.close()


def train_one_epoch(
        n_batches, train_file, ckpt_dir, run_dest_dir, learning_rate=0.01
):
    tf.reset_default_graph()

    with tf.Graph().as_default() as g:
        with tf.Session(graph=g) as sess:

            # set up model
            model = MNISTModel()
            with tf.variable_scope('input'):
                features, targets = batch_generator(
                    train_file, batch_size=128, num_epochs=1,
                    use_oned_data=True
                )

            # define all ops
            loss_op, accuracy_op = get_loss_and_acc_ops(
                model, features, targets
            )
            gstep_tensr = tf.train.get_or_create_global_step()
            with tf.variable_scope('training'):
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=learning_rate
                )
                train_op = optimizer.minimize(
                    loss_op, global_step=gstep_tensr
                )
            summary_op = create_full_summary_op(
                'summaries/train', loss_op, accuracy_op
            )

            # set up writer _after_ you've set up vars to be saved
            writer = tf.summary.FileWriter(run_dest_dir, graph=g)
            saver = tf.train.Saver(save_relative_paths=True)

            # variable initialization
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # checkpoint and state management
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                LOGGER.info('Restored session from {}'.format(ckpt_dir))
            writer.add_graph(sess.graph)

            # prep for training
            initial_step = gstep_tensr.eval()
            LOGGER.info('initial step = {}'.format(initial_step))

            # train
            try:
                for b_num in range(initial_step, initial_step + n_batches):
                    _, loss_batch, summary_t = sess.run(
                        [train_op, loss_op, summary_op]
                    )
                    LOGGER.info(
                        ' Loss @step {}: {:5.1f}'.format(b_num, loss_batch)
                    )
                    writer.add_summary(summary_t, global_step=b_num)
                    saver.save(sess, ckpt_dir, b_num)
            except tf.errors.OutOfRangeError:
                LOGGER.info('Training stopped - queue is empty.')

        writer.close()


def train(n_batches, train_file, valid_file, model_dir, learning_rate=0.01):

    n_epochs = 2
    for i in range(n_epochs):
        ckpt_dir, run_dest_dir = get_ckpt_and_run_dest(model_dir)
        train_one_epoch(
            n_batches, train_file, ckpt_dir, run_dest_dir, learning_rate
        )
        validate_one_epoch(valid_file, ckpt_dir, run_dest_dir)

    LOGGER.info('Finished training...')


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-n', '--n-batches', dest='n_batches',
                      help='Number of training batches', metavar='NBATCH',
                      default=20, type='int')
    parser.add_option('-m', '--model-dir', dest='model_dir',
                      help='Model directory', metavar='MODELDIR',
                      default='/tmp/logreg', type='string')
    parser.add_option('-t', '--train-file', dest='train_file',
                      help='Trail file', metavar='TRAINFILE',
                      default='train.dat', type='string')
    parser.add_option('-v', '--valid-file', dest='valid_file',
                      help='Validation file', metavar='VALIDFILE',
                      default='valid.dat', type='string')

    (options, args) = parser.parse_args()

    train(
        n_batches=options.n_batches,
        train_file=options.train_file,
        valid_file=options.valid_file,
        model_dir=options.model_dir,
    )
