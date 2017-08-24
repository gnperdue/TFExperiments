"""
MNIST models
"""
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier_init

import logging

LOGGER = logging.getLogger(__name__)


class MNISTLogReg:
    def __init__(self):
        self.loss = None
        self.logits = None
        self.global_step = None
        self.reg = tf.contrib.layers.l2_regularizer(scale=0.0001)
        # dropout not used here, but kept for API uniformity
        self.dropout_keep_prob = None

    def _create_summaries(self):
        base_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        with tf.name_scope('summaries/train'):
            train_loss = tf.summary.scalar('loss', self.loss)
            train_histo_loss = tf.summary.histogram(
                'histogram_loss', self.loss
            )
            train_reg_loss = tf.summary.scalar(
                'reg_loss', self.regularization_losses
            )
            train_accuracy = tf.summary.scalar('accuracy', self.accuracy)
            train_summaries = [
                train_loss, train_histo_loss, train_reg_loss, train_accuracy
            ]
            train_summaries.extend(base_summaries)
            self.train_summary_op = tf.summary.merge(train_summaries)
        with tf.name_scope('summaries/valid'):
            valid_loss = tf.summary.scalar('loss', self.loss)
            valid_histo_loss = tf.summary.histogram(
                'histogram_loss', self.loss
            )
            valid_reg_loss = tf.summary.scalar(
                'reg_loss', self.regularization_losses
            )
            valid_accuracy = tf.summary.scalar('accuracy', self.accuracy)
            valid_summaries = [
                valid_loss, valid_histo_loss, valid_reg_loss, valid_accuracy
            ]
            valid_summaries.extend(base_summaries)
            self.valid_summary_op = tf.summary.merge(valid_summaries)

    def _build_network(self, features):

        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name='dropout_keep_prob'
        )
        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step'
        )

        with tf.variable_scope('io'):
            self.X = features

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

    def _set_targets(self, targets):
        with tf.variable_scope('targets'):
            self.targets = tf.cast(targets, tf.float32)

    def _define_loss(self):
        with tf.variable_scope('loss'):
            self.regularization_losses = sum(
                tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES
                )
            )
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.targets
                ),
                axis=0,
                name='loss'
            ) + self.regularization_losses
            preds = tf.nn.softmax(self.logits, name='preds')
            correct_preds = tf.equal(
                tf.argmax(preds, 1), tf.argmax(self.targets, 1),
                name='correct_preds'
            )
            self.accuracy = tf.divide(
                tf.reduce_sum(tf.cast(correct_preds, tf.float32)),
                tf.cast(tf.shape(self.targets)[0], tf.float32),
                name='accuracy'
            )

    def _define_train_op(self, learning_rate):
        LOGGER.info('Building train op with learning_rate = %f' %
                    learning_rate)
        with tf.variable_scope('training'):
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate
            ).minimize(self.loss, global_step=self.global_step)

    def prepare_for_inference(self, features):
        self._build_network(features)

    def prepare_for_training(self, targets, learning_rate=0.001):
        self._set_targets(targets)
        self._define_loss()
        self._define_train_op(learning_rate)
        self._create_summaries()

    def prepare_for_loss_computation(self, targets):
        self._set_targets(targets)
        self._define_loss()
        self._create_summaries()

    def get_output_nodes(self):
        return ['model/logits']


class MNISTConvNet:
    def __init__(self):
        self.loss = None
        self.logits = None
        self.global_step = None
        self.reg = tf.contrib.layers.l2_regularizer(scale=0.0001)
        self.dropout_keep_prob = None

    def _create_summaries(self):
        base_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        with tf.name_scope('summaries/train'):
            train_loss = tf.summary.scalar('loss', self.loss)
            train_histo_loss = tf.summary.histogram(
                'histogram_loss', self.loss
            )
            train_accuracy = tf.summary.scalar('accuracy', self.accuracy)
            train_summaries = [train_loss, train_histo_loss, train_accuracy]
            train_summaries.extend(base_summaries)
            self.train_summary_op = tf.summary.merge(train_summaries)
        with tf.name_scope('summaries/valid'):
            valid_loss = tf.summary.scalar('loss', self.loss)
            valid_histo_loss = tf.summary.histogram(
                'histogram_loss', self.loss
            )
            valid_accuracy = tf.summary.scalar('accuracy', self.accuracy)
            valid_summaries = [valid_loss, valid_histo_loss, valid_accuracy]
            valid_summaries.extend(base_summaries)
            self.valid_summary_op = tf.summary.merge(valid_summaries)

    def _build_network(self, features):

        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name='dropout_keep_prob'
        )
        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step'
        )

        with tf.variable_scope('io'):
            self.X_img = tf.cast(features, tf.float32)

        with tf.variable_scope('model'):
            with tf.variable_scope('conv1'):
                self.kernels1 = tf.get_variable(
                    'kernels',
                    [5, 5, 1, 32],
                    initializer=xavier_init(uniform=False)
                )
                self.biases1 = tf.get_variable(
                    'biases',
                    [32],
                    initializer=xavier_init(uniform=False)
                )
                self.conv1 = tf.nn.conv2d(
                    self.X_img, self.kernels1, strides=[1, 1, 1, 1],
                    padding='SAME', data_format=self.data_format
                )
                self.conv1 = tf.nn.relu(
                    tf.nn.bias_add(
                        self.conv1, self.biases1, data_format=self.data_format
                    ),
                    name='relu'
                )

            with tf.variable_scope('pool1'):
                if self.data_format == 'NHWC':
                    ksize = [1, 2, 2, 1]
                    strides = [1, 2, 2, 1]
                elif self.data_format == 'NCHW':
                    ksize = [1, 1, 2, 2]
                    strides = [1, 1, 2, 2]
                else:
                    raise ValueError('Invalid data format!')
                self.pool1 = tf.nn.max_pool(
                    self.conv1, ksize=ksize, strides=strides,
                    padding='SAME', data_format=self.data_format,
                    name='pool1'
                )

            with tf.variable_scope('conv2'):
                self.kernels2 = tf.get_variable(
                    'kernels',
                    [5, 5, 32, 64],
                    initializer=xavier_init(uniform=False)
                )
                self.biases2 = tf.get_variable(
                    'biases',
                    [64],
                    initializer=xavier_init(uniform=False)
                )
                self.conv2 = tf.nn.conv2d(
                    self.pool1, self.kernels2, strides=[1, 1, 1, 1],
                    padding='SAME', data_format=self.data_format
                )
                self.conv2 = tf.nn.relu(
                    tf.nn.bias_add(
                        self.conv2, self.biases2, data_format=self.data_format
                    ),
                    name='relu'
                )

            with tf.variable_scope('pool2'):
                if self.data_format == 'NHWC':
                    ksize = [1, 2, 2, 1]
                    strides = [1, 2, 2, 1]
                elif self.data_format == 'NCHW':
                    ksize = [1, 1, 2, 2]
                    strides = [1, 1, 2, 2]
                else:
                    raise ValueError('Invalid data format!')
                self.pool2 = tf.nn.max_pool(
                    self.conv2, ksize=ksize, strides=strides,
                    padding='SAME', data_format=self.data_format,
                    name='pool2'
                )

            with tf.variable_scope('fc'):
                # use weight of dimension 7 * 7 * 64 x 1024
                input_features = 7 * 7 * 64
                self.weights_fc = tf.get_variable(
                    'weights',
                    [input_features, 1024],
                    initializer=xavier_init(uniform=False)
                )
                self.biases_fc = tf.get_variable(
                    'biases',
                    [1024],
                    initializer=xavier_init(uniform=False)
                )
                # reshape pool2 to 2 dimensional
                self.pool2 = tf.reshape(self.pool2, [-1, input_features])
                # apply relu on matmul of pool2 and w + b
                self.fc = tf.nn.relu(
                    tf.matmul(self.pool2, self.weights_fc) + self.biases_fc
                )
                # apply dropout
                self.fc = tf.nn.dropout(
                    self.fc, self.dropout_keep_prob, name='relu_dropout'
                )

            with tf.variable_scope('softmax_linear'):
                self.weights_softmax = tf.get_variable(
                    'weights',
                    [1024, self.n_classes],
                    initializer=xavier_init(uniform=False)
                )
                self.biases_softmax = tf.get_variable(
                    'biases',
                    [self.n_classes],
                    initializer=xavier_init(uniform=False)
                )
                self.logits = tf.add(
                    tf.matmul(self.fc, self.weights_softmax),
                    self.biases_softmax,
                    name='logits'
                )

    def _set_targets(self, targets):
        with tf.variable_scope('targets'):
            self.targets = tf.cast(targets, tf.float32)

    def _define_loss(self):
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.targets
                ),
                axis=0,
                name='loss'
            )
            preds = tf.nn.softmax(self.logits, name='preds')
            correct_preds = tf.equal(
                tf.argmax(preds, 1), tf.argmax(self.targets, 1),
                name='correct_preds'
            )
            self.accuracy = tf.divide(
                tf.reduce_sum(tf.cast(correct_preds, tf.float32)),
                tf.cast(tf.shape(self.targets)[0], tf.float32),
                name='accuracy'
            )

    def _define_train_op(self, learning_rate):
        LOGGER.info('Building train op with learning_rate = %f' %
                    learning_rate)
        with tf.name_scope('training'):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate
            ).minimize(self.loss, global_step=self.global_step)

    def prepare_for_inference(self, features):
        self._build_network(features)

    def prepare_for_training(self, targets, learning_rate=0.001):
        self._set_targets(targets)
        self._define_loss()
        self._define_train_op(learning_rate)
        self._create_summaries()

    def prepare_for_loss_computation(self, targets):
        self._set_targets(targets)
        self._define_loss()
        self._create_summaries()

    def get_output_nodes(self):
        return ['softmax_linear/logits']
