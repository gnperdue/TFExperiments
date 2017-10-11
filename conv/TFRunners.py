#!/usr/bin/env python
"""
Run TF
"""
from __future__ import print_function
import os
import time
import logging

import tensorflow as tf
import numpy as np
from six.moves import range

from DataReaders import MNISTDataReader
import utils_mnist

LOGGER = logging.getLogger(__name__)


class TFRunnerCategorical:
    """
    runner class for categorical classification
    (not sure we need to make this distinction here)
    """
    def __init__(
            self,
            model,
            run_params_dict,
            train_params_dict=None
    ):
        if train_params_dict is None:
            train_params_dict = dict()

        try:
            self.train_file_list = run_params_dict['TRAIN_FILE_LIST']
            self.valid_file_list = run_params_dict['VALID_FILE_LIST']
            self.test_file_list = run_params_dict['TEST_FILE_LIST']
            self.file_compression = run_params_dict['COMPRESSION']
            self.save_model_directory = run_params_dict['MODEL_DIR']
            self.load_saved_model = run_params_dict['LOAD_SAVED_MODEL']
            self.save_freq = run_params_dict['SAVE_EVRY_N_BATCHES']
            self.be_verbose = run_params_dict['BE_VERBOSE']
        except KeyError as e:
            print(e)

        self.learning_rate = train_params_dict.get('LEARNING_RATE', 0.001)
        self.batch_size = train_params_dict.get('BATCH_SIZE', 128)
        self.num_epochs = train_params_dict.get('NUM_EPOCHS', 1)
        self.momentum = train_params_dict.get('MOMENTUM', 0.9)
        self.dropout_keep_prob = train_params_dict.get(
            'DROPOUT_KEEP_PROB', 0.75
        )
        self.strategy = train_params_dict.get(
            'STRATEGY', tf.train.AdamOptimizer
        )

        self.model = model

    def _prep_targets_and_features(self, generator, num_epochs):
        batch_dict = generator(num_epochs=num_epochs)
        targets = batch_dict['targets']
        features = batch_dict['features']
        return targets, features

    def run_training(
            self, do_validation=False, short=False, is_image=False
    ):
        """
        run training (TRAIN file list) and optionally run a validation pass
        (on the VALID file list)
        """
        LOGGER.info('staring run_training, image = {}...'.format(is_image))
        tf.reset_default_graph()
        initial_batch = 0
        ckpt_dir = self.save_model_directory + '/checkpoints'
        run_dest_dir = self.save_model_directory + '/%d' % time.time()
        LOGGER.info('tensorboard command:')
        LOGGER.info('\ttensorboard --logdir {}'.format(
            self.save_model_directory
        ))

        with tf.Graph().as_default() as g:

            # n_batches: control this with num_epochs
            n_batches = 100 if short else int(1e9)
            save_every_n_batch = 10 if short else self.save_freq
            LOGGER.info(' Processing {} batches, saving every {}...'.format(
                n_batches, save_every_n_batch
            ))

            with tf.Session(graph=g) as sess:

                with tf.variable_scope('data_io'):
                    train_reader = MNISTDataReader(
                        filenames_list=self.train_file_list,
                        batch_size=self.batch_size,
                        name='train',
                        compression=self.file_compression,
                        is_image=is_image
                    )
                    targets_train, features_train = \
                        self._prep_targets_and_features(
                            train_reader.shuffle_batch_generator,
                            self.num_epochs
                        )

                    valid_reader = MNISTDataReader(
                        filenames_list=self.valid_file_list,
                        batch_size=self.batch_size,
                        name='valid',
                        compression=self.file_compression,
                        is_image=is_image
                    )
                    targets_valid, features_valid = \
                        self._prep_targets_and_features(
                            valid_reader.batch_generator,
                            1000000
                        )

                    with tf.variable_scope('pipeline_control'):
                        use_valid = tf.placeholder(
                            tf.bool, shape=(), name='train_val_batch_logic'
                        )

                def get_features_train():
                    return features_train

                def get_features_valid():
                    return features_valid

                def get_targets_train():
                    return targets_train

                def get_targets_valid():
                    return targets_valid

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

                self.model.prepare_for_inference(features)
                self.model.prepare_for_training(
                    targets, learning_rate=self.learning_rate
                )
                LOGGER.info('Preparing to train model with %d parameters' %
                            utils_mnist.get_number_of_trainable_parameters())

                writer = tf.summary.FileWriter(run_dest_dir)
                saver = tf.train.Saver()

                start_time = time.time()
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    LOGGER.info('Restored session from {}'.format(ckpt_dir))

                writer.add_graph(sess.graph)
                initial_batch = self.model.global_step.eval()
                LOGGER.info('initial step = %d' % initial_batch)

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                # NOTE: specifically catch `tf.errors.OutOfRangeError` or we
                # won't handle the exception correctly.
                try:
                    for b_num in range(
                            initial_batch, initial_batch + n_batches
                    ):
                        LOGGER.debug('  processing batch {}'.format(b_num))
                        _, loss, summary_t = sess.run(
                            [self.model.optimizer,
                             self.model.loss,
                             self.model.train_summary_op],
                            feed_dict={
                                use_valid: False,
                                self.model.dropout_keep_prob:
                                self.dropout_keep_prob,
                                self.model.is_training: True
                            }
                        )
                        writer.add_summary(summary_t, global_step=b_num)
                        LOGGER.info(
                            '  Train loss at batch {}: {:5.1f}'.format(
                                b_num, loss
                            )
                        )
                        if (b_num + 1) % save_every_n_batch == 0:
                            # validation
                            loss, accuracy, logits, targs, summary_v = \
                                sess.run(
                                    [self.model.loss,
                                     self.model.accuracy,
                                     self.model.logits,
                                     self.model.targets,
                                     self.model.valid_summary_op],
                                    feed_dict={
                                        use_valid: True,
                                        self.model.dropout_keep_prob: 1.0,
                                        self.model.is_training: False
                                    }
                                )
                            saver.save(sess, ckpt_dir, b_num)
                            writer.add_summary(summary_v, global_step=b_num)
                            preds = tf.nn.softmax(logits)
                            LOGGER.info('   preds   = \n{}'.format(
                                tf.argmax(preds, 1).eval()
                            ))
                            LOGGER.info('   Y_batch = \n{}'.format(
                                np.argmax(targs, 1)
                            ))
                            LOGGER.info('    accuracy = {}'.format(
                                accuracy
                            ))
                            LOGGER.info(
                                '  Valid loss at batch {}: {:5.1f}'.format(
                                    b_num, loss
                                )
                            )
                            LOGGER.info('   Elapsed time = {}'.format(
                                time.time() - start_time
                            ))
                except tf.errors.OutOfRangeError:
                    LOGGER.info('Training stopped - queue is empty.')
                    LOGGER.info(
                        'Executing final save at batch {}'.format(b_num)
                    )
                    saver.save(sess, ckpt_dir, b_num)
                except Exception as e:
                    LOGGER.error(e)
                finally:
                    coord.request_stop()
                    coord.join(threads)

            writer.close()

        out_graph = utils_mnist.freeze_graph(
            self.save_model_directory, self.model.get_output_nodes()
        )
        LOGGER.info(' Saved graph {}'.format(out_graph))
        LOGGER.info('Finished training...')

    def run_testing(self, short=False, is_image=False):
        """
        run a test pass (not "validation"!), based on the TEST file list.
        """
        LOGGER.info('Starting testing...')
        tf.reset_default_graph()
        ckpt_dir = self.save_model_directory + '/checkpoints'

        with tf.Graph().as_default() as g:

            n_batches = 2 if short else int(1e9)
            LOGGER.info(' Processing {} batches...'.format(n_batches))

            with tf.Session(graph=g) as sess:
                with tf.variable_scope('data_io'):
                    data_reader = MNISTDataReader(
                        filenames_list=self.test_file_list,
                        batch_size=self.batch_size,
                        name='test',
                        compression=self.file_compression,
                        is_image=is_image
                    )
                    targets, features = \
                        self._prep_targets_and_features(
                            data_reader.batch_generator, 1
                        )

                self.model.prepare_for_inference(features)
                self.model.prepare_for_loss_computation(targets)
                LOGGER.info('Preparing to test model with %d parameters' %
                            utils_mnist.get_number_of_trainable_parameters())

                saver = tf.train.Saver()
                start_time = time.time()

                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    LOGGER.info('Restored session from {}'.format(ckpt_dir))

                final_step = self.model.global_step.eval()
                LOGGER.info('evaluation after {} steps.'.format(final_step))
                average_loss = 0.0
                total_correct_preds = 0

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                # NOTE: specifically catch `tf.errors.OutOfRangeError` or we
                # won't handle the exception correctly.
                n_processed = 0
                try:
                    for i in range(n_batches):
                        loss_batch, logits_batch, Y_batch = sess.run(
                            [self.model.loss, self.model.logits, targets],
                            feed_dict={
                                self.model.dropout_keep_prob: 1.0,
                                self.model.is_training: False
                            }
                        )
                        batch_sz = logits_batch.shape[0]
                        n_processed += batch_sz
                        average_loss += loss_batch
                        preds = tf.nn.softmax(logits_batch)
                        correct_preds = tf.equal(
                            tf.argmax(preds, 1), tf.argmax(Y_batch, 1)
                        )
                        if self.be_verbose:
                            LOGGER.debug('   preds   = \n{}'.format(
                                tf.argmax(preds, 1).eval()
                            ))
                            LOGGER.debug('   Y_batch = \n{}'.format(
                                tf.argmax(Y_batch, 1).eval()
                            ))
                        accuracy = tf.reduce_sum(
                            tf.cast(correct_preds, tf.float32)
                        )
                        total_correct_preds += sess.run(accuracy)
                        if self.be_verbose:
                            LOGGER.debug(
                                '  batch {} loss = {} for size = {}'.format(
                                    i, loss_batch, batch_sz
                                )
                            )

                except tf.errors.OutOfRangeError:
                    LOGGER.info('Testing stopped - queue is empty.')
                except Exception as e:
                    LOGGER.error(e)
                finally:
                    if n_processed > 0:
                        LOGGER.info("n_processed = {}".format(n_processed))
                        LOGGER.info(
                            " Total correct preds = {}".format(
                                total_correct_preds
                            )
                        )
                        LOGGER.info("  Accuracy: {}".format(
                            total_correct_preds / n_processed
                        ))
                        LOGGER.info('  Average loss: {}'.format(
                            average_loss / n_processed
                        ))
                    coord.request_stop()
                    coord.join(threads)

            LOGGER.info('  Elapsed time = {}'.format(time.time() - start_time))

        LOGGER.info('Finished testing...')
