"""
Simple logistic regression to test TF framework APIs.

MNIST TF records available here:
    https://github.com/gnperdue/RandomData/tree/master/TensorFlow

Graph freezing and loading functions inspired by:
    https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
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
        with tf.name_scope('summaries/valid'):
            valid_loss = tf.summary.scalar('loss', self.loss)
            valid_histo_loss = tf.summary.histogram(
                'histogram_loss', self.loss
            )
            valid_summaries = [valid_loss, valid_histo_loss]
            valid_summaries.extend(base_summaries)
            self.valid_summary_op = tf.summary.merge(valid_summaries)

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


def freeze_graph(
        model_dir, output_nodes_list, output_graph_name='frozen_model.pb'
):
    """
    reduce a saved model and metadata down to a deployable file
    """
    from tensorflow.python.framework import graph_util

    LOGGER.info('Attempting to freeze graph at {}'.format(model_dir))
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    if input_checkpoint is None:
        LOGGER.error('Cannot load checkpoint at {}'.format(model_dir))
        return None

    absolute_model_dir = '/'.join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + '/' + output_graph_name
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, input_graph_def, output_nodes_list
        )
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        LOGGER.info('Froze graph with {} ops'.format(
            len(output_graph_def.node)
        ))

    return output_graph


def load_frozen_graph(graph_filename):
    """
    load a protobuf *into the default graph* and parse it
    """
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    tf.import_graph_def(
        graph_def, input_map=None, return_elements=None, name='',
        op_dict=None, producer_op_list=None
    )


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
            features_train, targets_train = batch_generator(
                [train_file], stage_name='train', batch_size=128, num_epochs=1
            )
            valid_file = data_dir + 'mnist_valid.tfrecord.gz'
            features_valid, targets_valid = batch_generator(
                [valid_file], stage_name='valid', batch_size=128, num_epochs=1000
            )

            def get_features_train():
                return features_train

            def get_features_valid():
                return features_valid

            def get_targets_train():
                return targets_train

            def get_targets_valid():
                return targets_valid

            with tf.variable_scope('pipeline_control'):
                use_valid = tf.placeholder(
                    tf.bool, shape=(), name='train_val_batch_logic'
                )

            features = tf.cond(
                use_valid,
                get_features_valid,
                get_features_train,
                name='features_selection'
            )
            targets = tf.cond(
                use_valid,
                get_targets_valid,
                get_targets_train,
                name='targets_selection'
            )
            model.build_network(features, targets)
            writer = tf.summary.FileWriter(run_dest_dir)
            saver = tf.train.Saver()

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
                        [model.optimizer, model.loss, model.train_summary_op],
                        feed_dict={use_valid: False}
                    )
                    LOGGER.info(
                        ' Loss @step {}: {:5.1f}'.format(b_num, loss_batch)
                    )
                    writer.add_summary(summary_t, global_step=b_num)
                    if (b_num + 1) % 5 == 0:
                        # validation
                        loss_batch, summary_v = \
                            sess.run(
                                [model.loss,
                                 model.valid_summary_op],
                                feed_dict={use_valid: True}
                            )
                        LOGGER.info(
                            '  Valid loss @step {}: {:5.1f}'.format(
                                b_num, loss_batch
                            )
                        )
                        saver.save(sess, ckpt_dir, b_num)
                        writer.add_summary(summary_v, global_step=b_num)
            except tf.errors.OutOfRangeError:
                LOGGER.info('Training stopped - queue is empty.')
            except Exception as e:
                LOGGER.error(e)
            finally:
                coord.request_stop()
                coord.join(threads)

        writer.close()

    out_graph = freeze_graph(
        model_dir, ['model/logits']
    )
    LOGGER.info('Saved graph to {}'.format(out_graph))
    LOGGER.info('Finished training...')


def test_ckpt(n_batches, model_dir, data_dir, batch_size=10):
    """ test via checkpoint - always works on the train machine, fails elsewhere """
    tf.reset_default_graph()
    LOGGER.info('Starting testing via checkpoint...')

    ckpt_dir = model_dir + '/checkpoints'

    g = tf.Graph()
    with g.as_default():

        with tf.Session(graph=g) as sess:

            model = MNISTLogReg()
            test_file = data_dir + 'mnist_test.tfrecord.gz'
            features_batch, targets_batch = batch_generator(
                [test_file], stage_name='test', batch_size=batch_size, num_epochs=1
            )

            model.build_network(features_batch, targets_batch)

            saver = tf.train.Saver()

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                LOGGER.info('Restored session from {}'.format(ckpt_dir))

            average_loss = 0.0
            total_correct_preds = 0

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            n_processed = 0
            try:
                for i in range(n_batches):
                    loss_batch, logits_batch, Y_batch = sess.run(
                        [model.loss, model.logits, targets_batch]
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
            except Exception as e:
                LOGGER.error(e)
            finally:
                coord.request_stop()
                coord.join(threads)

    print('Finished testing...')
    LOGGER.info('Finished testing...')


def test_pb(n_batches, model_dir, data_dir, batch_size=10):
    """ attempt to use a frozen protobuf to do the test """
    tf.reset_default_graph()
    LOGGER.info('Starting testing via checkpoint...')

    g = tf.Graph()
    with g.as_default():
        load_frozen_graph(model_dir + '/frozen_model.pb')
        for tnsr in g.as_graph_def().node:
            LOGGER.debug('tnsr name = {}'.format(tnsr.name))

        with tf.Session(graph=g) as sess:

            test_file = data_dir + 'mnist_test.tfrecord.gz'
            features_batch, targets_batch = batch_generator(
                [test_file], stage_name='train', batch_size=batch_size, num_epochs=1
            )
            with tf.variable_scope('pipeline_control'):
                use_valid = tf.placeholder(
                    tf.bool, shape=(), name='train_val_batch_logic'
                )

            logits = g.get_tensor_by_name('model/logits:0')

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            total_correct_preds = 0

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            n_processed = 0
            try:
                for i in range(n_batches):
                    logits_batch, Y_batch = sess.run(
                        [logits, targets_batch],
                        feed_dict={use_valid: False}
                    )
                    n_processed += batch_size
                    preds = tf.nn.softmax(logits_batch)
                    correct_preds = tf.equal(
                        tf.argmax(preds, 1), tf.argmax(Y_batch, 1)
                    )
                    accuracy = tf.reduce_sum(
                        tf.cast(correct_preds, tf.float32)
                    )
                    total_correct_preds += sess.run(accuracy)
                    LOGGER.info("  total_corr_preds / nproc = {} / {}".format(
                        total_correct_preds, n_processed
                    ))
                    LOGGER.info("  Cumul. Accuracy {0}".format(
                        total_correct_preds / n_processed
                    ))
            except tf.errors.OutOfRangeError:
                LOGGER.info('Testing stopped - queue is empty.')
            except Exception as e:
                LOGGER.error(e)
            finally:
                coord.request_stop()
                coord.join(threads)

    print('Finished testing via load_frozen_graph...')
    LOGGER.info('Finished testing via load_frozen_graph...')


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

    # train(
    #     n_batches=options.n_batches,
    #     model_dir=options.model_dir,
    #     data_dir=options.data_dir
    # )
    test_ckpt(
        n_batches=10,
        model_dir=options.model_dir,
        data_dir=options.data_dir
    )
    # test_pb(
    #     n_batches=10,
    #     model_dir=options.model_dir,
    #     data_dir=options.data_dir
    # )
