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

tf.logging.set_verbosity(tf.logging.DEBUG)

logging.basicConfig(
    filename='tmp_logreg_log.txt', level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def train(n_batches, train_file, model_dir):
    tf.reset_default_graph()
    ckpt_dir = model_dir + '/checkpoints'
    run_dest_dir = model_dir + '/%d' % time.time()

    with tf.Graph().as_default() as g:

        with tf.Session(graph=g) as sess:

            model = MNISTModel()
            with tf.variable_scope('input'):
                features, targets = batch_generator(
                    train_file, batch_size=128, num_epochs=1,
                    use_oned_data=True
                )
            model.build_network(features, targets)
            writer = tf.summary.FileWriter(run_dest_dir)
            saver = tf.train.Saver(save_relative_paths=True)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                LOGGER.info('Restored session from {}'.format(ckpt_dir))

            writer.add_graph(sess.graph)
            initial_step = model.global_step.eval()
            LOGGER.info('initial step = {}'.format(initial_step))

            try:
                for b_num in range(initial_step, initial_step + n_batches):
                    _, loss_batch, summary_t = sess.run(
                        [model.optimizer, model.loss, model.train_summary_op]
                    )
                    LOGGER.info(
                        ' Loss @step {}: {:5.1f}'.format(b_num, loss_batch)
                    )
                    writer.add_summary(summary_t, global_step=b_num)
                    saver.save(sess, ckpt_dir, b_num)
            except tf.errors.OutOfRangeError:
                LOGGER.info('Training stopped - queue is empty.')

        writer.close()

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

    (options, args) = parser.parse_args()

    train(
        n_batches=options.n_batches,
        model_dir=options.model_dir,
        train_file=options.train_file
    )
