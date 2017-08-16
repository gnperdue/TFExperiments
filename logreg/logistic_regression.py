import tensorflow as tf
import numpy as np

import time
import os
import logging

from model_mnist import MNISTLogReg
from tfrecord_mnist import batch_generator
from tfrecord_mnist import DATA_PATH
import utils_mnist

logging.basicConfig(
    filename='tmp_logreg_log.txt', level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)

TBOARD_DEST_DIR = '/tmp/logreg'


def train():
    tf.reset_default_graph()
    ckpt_dir = TBOARD_DEST_DIR + '/checkpoints'
    run_dest_dir = TBOARD_DEST_DIR + '/%d' % time.time()
    print('tensorboard command:')
    print('\ttensorboard --logdir {}'.format(TBOARD_DEST_DIR))

    with tf.Graph().as_default() as g:

        with tf.Session(graph=g) as sess:

            model = MNISTLogReg()
            train_file = DATA_PATH + 'mnist_train.tfrecord.gz'
            features_batch, targets_batch = batch_generator(
                [train_file], batch_size=128, num_epochs=1
            )
        
            model.build_network(features_batch, targets_batch)
            writer = tf.summary.FileWriter(run_dest_dir)
            saver = tf.train.Saver()

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            writer.add_graph(sess.graph)
            average_loss = 0.0

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            
            try:
                for b_num in range(0, 20):
                    _, loss_batch = sess.run([model.optimizer, model.loss])
                    average_loss += loss_batch
                    if (b_num + 1) % 5 == 0:
                        LOGGER.info('  Avg loss at step {}: {:5.1f}'.format(
                            b_num + 1, average_loss / 5
                        ))
                        average_loss = 0.0
                        saver.save(sess, ckpt_dir, b_num)
                        LOGGER.info('     saved at iter %d' % b_num)
                        LOGGER.debug(' Weights [0, :10] = {}'.format(
                            g.get_tensor_by_name('model/weights:0').eval()[0, :10]
                        ))
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


def test_ckpt():
    tf.reset_default_graph()
    LOGGER.info('Starting testing via checkpoint...')

    ckpt_dir = TBOARD_DEST_DIR + '/checkpoints'

    with tf.Graph().as_default() as g:

        with tf.Session(graph=g) as sess:

            model = MNISTLogReg()
            test_file = DATA_PATH + 'mnist_test.tfrecord.gz'
            features_batch, targets_batch = batch_generator(
                [test_file], batch_size=10, num_epochs=1
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

            LOGGER.debug(' Weights [0, :10] = {}'.format(
                g.get_tensor_by_name('model/weights:0').eval()[0, :10]
            ))

            average_loss = 0.0
            total_correct_preds = 0

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            n_processed = 0
            try:
                for i in range(5):
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


def test_proto():
    tf.reset_default_graph()
    LOGGER.info('Starting testing via saved protobuf...')

    with tf.Graph().as_default() as g:

        with tf.Session(graph=g) as sess:

            model = MNISTLogReg()
            test_file = DATA_PATH + 'mnist_test.tfrecord.gz'
            features_batch, targets_batch = batch_generator(
                [test_file], batch_size=10, num_epochs=1
            )
            model.build_network(features_batch, targets_batch)
            utils_mnist.load_frozen_graph(TBOARD_DEST_DIR + '/frozen_model.pb')

            for tnsr in g.as_graph_def().node:
                LOGGER.debug(' tnsr name = {}'.format(tnsr.name))

            LOGGER.info('Preparing to test model with %d parameters' %
                        utils_mnist.get_number_of_trainable_parameters())

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            LOGGER.debug(' Weights [0, :10] = {}'.format(
                g.get_tensor_by_name('model/weights:0').eval()[0, :10]
            ))

    print('Finished testing via saved protobuf...')
    LOGGER.info('Finished testing via saved protobuf...')
        

if __name__ == '__main__':
    train()
    test_ckpt()
    test_proto()



