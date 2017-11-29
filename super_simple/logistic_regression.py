"""
Super-Simple logistic regression to test TF framework APIs.

MNIST TF records available here:
    https://github.com/gnperdue/RandomData/tree/master/TensorFlow

Sample run script:
```
#!/bin/bash

NTRAINBATCH=20
if [ $# -gt 0 ]; then
  NTRAINBATCH=$1
fi

DATADIR="${HOME}//Dropbox/Data/RandomData/TensorFlow/"
MODELDIR="/tmp/models"

python logistic_regression.py -n $NTRAINBATCH -m $MODELDIR -d $DATADIR
```
"""
import tensorflow as tf

import time
import os
import logging

logging.basicConfig(
    filename='tmp_logreg_log.txt', level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


class MNISTLogReg:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.loss = None
        self.logits = None
        self.global_step = None
        self.reg = tf.contrib.layers.l2_regularizer(scale=0.0001)

    def _create_summaries(self):
        base_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        with tf.name_scope('summaries/train'):
            train_loss = tf.summary.scalar('loss', self.loss)
            train_histo_loss = tf.summary.histogram(
                'histogram_loss', self.loss
            )
            train_summaries = [train_loss, train_histo_loss]
            train_summaries.extend(base_summaries)
            self.train_summary_op = tf.summary.merge(train_summaries)

    def build_network(self, features, targets):

        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step'
        )

        with tf.variable_scope('io'):
            self.X = features
            self.Y = targets

        with tf.variable_scope('model'):
            self.W = tf.get_variable(
                name='weights',
                initializer=tf.random_normal(
                    shape=[784, 10], mean=0.0, stddev=0.01, dtype=tf.float32
                ),
                regularizer=self.reg
            )
            self.b = tf.get_variable(
                name='bias',
                initializer=tf.random_normal(
                    shape=[10], mean=0.0, stddev=0.01, dtype=tf.float32
                ),
                regularizer=self.reg
            )
            self.logits = tf.add(
                tf.matmul(self.X, self.W), self.b, name='logits'
            )

        with tf.variable_scope('loss'):
            self.regularization_losses = sum(
                tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES
                )
            )
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.Y
                ),
                axis=0,
                name='loss'
            ) + self.regularization_losses

        with tf.variable_scope('training'):
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.learning_rate
            ).minimize(self.loss, global_step=self.global_step)

        self._create_summaries()


def tfrecord_to_graph_ops(filenames_list, num_epochs=1):
    with tf.variable_scope('tfrec_to_graph'):
        file_queue = tf.train.string_input_producer(
            filenames_list, name='file_queue', num_epochs=num_epochs
        )
        reader = tf.TFRecordReader(
            options=tf.python_io.TFRecordOptions(
                compression_type=tf.python_io.TFRecordCompressionType.GZIP
            ), name='tfrec_reader'
        )
        _, tfrecord = reader.read(file_queue)

        tfrecord_features = tf.parse_single_example(
            tfrecord,
            features={
                'features': tf.FixedLenFeature([], tf.string),
                'targets': tf.FixedLenFeature([], tf.string)
            },
            name='data'
        )
        with tf.variable_scope('features'):
            features = tf.decode_raw(tfrecord_features['features'], tf.uint8)
            features = tf.reshape(features, [-1, 784])
            features = tf.cast(features, tf.float32)
        with tf.variable_scope('targets'):
            targets = tf.decode_raw(tfrecord_features['targets'], tf.uint8)
            targets = tf.cast(targets, tf.int32)
            targets = tf.reshape(targets, [-1])
            targets = tf.one_hot(
                indices=targets, depth=10, on_value=1, off_value=0
            )
        return features, targets


def batch_generator(
        filenames_list, stage_name='train', batch_size=64, num_epochs=1
):
    with tf.variable_scope(stage_name + '/batchgen'):
        features, targets = tfrecord_to_graph_ops(filenames_list, num_epochs)
        min_after_dequeue = 3 * batch_size
        capacity = 20 * batch_size
        features_batch, targets_batch = tf.train.shuffle_batch(
            [features, targets],
            batch_size=batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            enqueue_many=True
        )
    return features_batch, targets_batch


def train(n_batches, model_dir, data_dir):
    tf.reset_default_graph()
    ckpt_dir = model_dir + '/checkpoints'
    run_dest_dir = model_dir + '/%d' % time.time()

    with tf.Graph().as_default() as g:

        with tf.Session(graph=g) as sess:

            model = MNISTLogReg()
            train_file = data_dir + 'mnist_train.tfrecord.gz'
            features, targets = batch_generator(
                [train_file], stage_name='train', batch_size=128, num_epochs=1
            )
            model.build_network(features, targets)
            writer = tf.summary.FileWriter(run_dest_dir)
            saver = tf.train.Saver(
                save_relative_paths=True
            )

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                LOGGER.info('Restored session from {}'.format(ckpt_dir))

            writer.add_graph(sess.graph)
            initial_step = model.global_step.eval()
            LOGGER.info('initial step = {}'.format(initial_step))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                for b_num in range(initial_step, initial_step + n_batches):
                    # regular training
                    _, loss_batch, summary_t = sess.run(
                        [model.optimizer, model.loss, model.train_summary_op]
                    )
                    LOGGER.info(
                        ' Loss @step {}: {:5.1f}'.format(b_num, loss_batch)
                    )
                    writer.add_summary(summary_t, global_step=b_num)
            except tf.errors.OutOfRangeError:
                LOGGER.info('Training stopped - queue is empty.')
            except Exception as e:
                LOGGER.error(e)
            finally:
                coord.request_stop()
                coord.join(threads)

        writer.close()

    LOGGER.info('Finished training...')


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-n', '--n_batches', dest='n_batches',
                      help='Number of training batches', metavar='NBATCH',
                      default=20, type='int')
    parser.add_option('-m', '--model_dir', dest='model_dir',
                      help='Model directory', metavar='MODELDIR',
                      default='/tmp/logreg', type='string')
    parser.add_option('-d', '--data_dir', dest='data_dir',
                      help='Data directory', metavar='DATADIR',
                      default='/tmp/data', type='string')

    (options, args) = parser.parse_args()

    train(
        n_batches=options.n_batches,
        model_dir=options.model_dir,
        data_dir=options.data_dir
    )
