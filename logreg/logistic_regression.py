"""
Simple logistic regression to test TF framework APIs.
"""
import tensorflow as tf
import numpy as np

import time
import os
import logging

from model_mnist import MNISTLogReg
from tfrecord_mnist import batch_generator
import utils_mnist

logging.basicConfig(
    filename='tmp_logreg_log.txt', level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)

TBOARD_DEST_DIR = '/tmp/logreg'


def train_tfrec(n_batches):
    from tfrecord_mnist import DATA_PATH

    tf.reset_default_graph()
    ckpt_dir = TBOARD_DEST_DIR + '/checkpoints'
    run_dest_dir = TBOARD_DEST_DIR + '/%d' % time.time()
    print('tensorboard command:')
    print('\ttensorboard --logdir {}'.format(TBOARD_DEST_DIR))

    with tf.Graph().as_default() as g:

        with tf.Session(graph=g) as sess:

            model = MNISTLogReg()
            train_file = DATA_PATH + 'mnist_train.tfrecord.gz'
            features_train, targets_train = batch_generator(
                [train_file], stage_name='train', batch_size=128, num_epochs=1
            )
            valid_file = DATA_PATH + 'mnist_valid.tfrecord.gz'
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

            cntr = tf.placeholder(tf.int32, shape=(), name='batch_counter')
            pfrq = tf.constant(5, dtype=tf.int32, name='const_val_mod_nmbr')
            tfzo = tf.constant(0, dtype=tf.int32, name='const_zero')
            pred = tf.equal(tf.mod(cntr, pfrq), tfzo, name='train_valid_pred')
            features = tf.cond(
                pred,
                get_features_train,
                get_features_valid,
                name='features_selection'
            )
            targets = tf.cond(
                pred,
                get_targets_train,
                get_targets_valid,
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
                    if (b_num + 1) % 5 == 0:
                        # validation
                        loss_batch, logits_batch, Y_batch, summary = sess.run(
                            [model.loss, model.logits, model.Y, model.valid_summary_op],
                            feed_dict={cntr: (b_num + 1)}
                        )
                        LOGGER.info(
                            '  Valid loss @step {}: {:5.1f}'.format(
                                b_num, loss_batch
                            )
                        )
                        preds = tf.nn.softmax(logits_batch)
                        correct_preds = tf.equal(
                            tf.argmax(preds, 1), tf.argmax(Y_batch, 1)
                        )
                        LOGGER.info('   preds   = \n{}'.format(
                            tf.argmax(preds, 1).eval()
                        ))
                        LOGGER.info('   Y_batch = \n{}'.format(
                            np.argmax(Y_batch, 1)
                        ))
                        accuracy = tf.reduce_sum(
                            tf.cast(correct_preds, tf.float32)
                        )
                        LOGGER.info('    accuracy = {}'.format(
                            accuracy.eval() / Y_batch.shape[0]
                        ))
                        LOGGER.debug('    Weights [300, :10] = {}'.format(
                            g.get_tensor_by_name(
                                'model/weights:0'
                            ).eval()[300, :10]
                        ))
                        saver.save(sess, ckpt_dir, b_num)
                        writer.add_summary(summary, global_step=b_num)
                    else:
                        # regular training
                        _, loss_batch, summary = sess.run(
                            [model.optimizer, model.loss, model.train_summary_op],
                            feed_dict={cntr: (b_num + 1)}
                        )
                        LOGGER.info(
                            ' Loss @step {}: {:5.1f}'.format(
                                b_num, loss_batch
                            )
                        )
                        writer.add_summary(summary, global_step=b_num)
            except tf.errors.OutOfRangeError:
                LOGGER.info('Training stopped - queue is empty.')
            except Exception as e:
                LOGGER.error(e)
            finally:
                coord.request_stop()
                coord.join(threads)

        writer.close()

    out_graph = utils_mnist.freeze_graph(
        TBOARD_DEST_DIR, ['model/logits']
    )
    LOGGER.info('Saved graph to {}'.format(out_graph))
    print('Finished training...')
    LOGGER.info('Finished training...')


def test_tfrec(n_batches=5):
    from tfrecord_mnist import DATA_PATH

    tf.reset_default_graph()
    LOGGER.info('Starting testing via checkpoint...')

    ckpt_dir = TBOARD_DEST_DIR + '/checkpoints'

    with tf.Graph().as_default() as g:

        with tf.Session(graph=g) as sess:

            model = MNISTLogReg()
            test_file = DATA_PATH + 'mnist_test.tfrecord.gz'
            features_batch, targets_batch = batch_generator(
                [test_file], stage_name='test', batch_size=10, num_epochs=1
            )

            model.build_network(features_batch, targets_batch)

            saver = tf.train.Saver()

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                LOGGER.info('Restored session from {}'.format(ckpt_dir))
                for tnsr in g.as_graph_def().node:
                    LOGGER.debug(' tnsr name = {}'.format(tnsr.name))

            LOGGER.debug(' Weights [300, :10] = {}'.format(
                g.get_tensor_by_name('model/weights:0').eval()[300, :10]
            ))

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
                    n_processed += 10
                    average_loss += loss_batch
                    preds = tf.nn.softmax(logits_batch)
                    correct_preds = tf.equal(
                        tf.argmax(preds, 1), tf.argmax(Y_batch, 1)
                    )
                    LOGGER.info('   preds   = \n{}'.format(
                        tf.argmax(preds, 1).eval()
                    ))
                    LOGGER.info('   Y_batch = \n{}'.format(
                        np.argmax(Y_batch, 1)
                    ))
                    accuracy = tf.reduce_sum(
                        tf.cast(correct_preds, tf.float32)
                    )
                    total_correct_preds += sess.run(accuracy)
                    LOGGER.info('  batch {} loss = {} for nproc {}'.format(
                        i, loss_batch, n_processed
                    ))

                    LOGGER.info("  total_correct_preds = {}".format(
                        total_correct_preds
                    ))
                    LOGGER.info("  n_processed = {}".format(
                        n_processed
                    ))
                    LOGGER.info(" Accuracy {0}".format(
                        total_correct_preds / n_processed
                    ))
            except tf.errors.OutOfRangeError:
                LOGGER.info('Testing stopped - queue is empty.')
            except Exception as e:
                LOGGER.error(e)
            finally:
                coord.request_stop()
                coord.join(threads)

    print('Finished testing via checkpoint...')
    LOGGER.info('Finished testing via checkpoint...')
        

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-n', '--n_batches', dest='n_batches',
                      help='Number of training batches', metavar='NBATCH',
                      default=20, type='int')

    (options, args) = parser.parse_args()

    train_tfrec(n_batches=options.n_batches)
    test_tfrec(n_batches=10)
