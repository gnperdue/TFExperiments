"""
MNIST models
"""
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier_init

import logging

LOGGER = logging.getLogger(__name__)


class LayerCreator:
    def __init__(
            self, regularization_strategy='l2', regularization_scale=0.0001
    ):
        if regularization_strategy == 'l2':
            self.reg = tf.contrib.layers.l2_regularizer(
                scale=regularization_scale
            )
        else:
            raise NotImplementedError(
                'Regularization strategy ' + regularization_strategy + ' \
                is not implemented yet.'
            )

        def make_wbkernels(
                name, shp, initializer=xavier_init(uniform=False),
        ):
            """ make weights, biases, kernels """
            return tf.get_variable(
                name, shp, initializer=initializer, regularizer=self.reg
            )
        

class MNISTLogReg:
    def __init__(self):
        self.loss = None
        self.logits = None
        self.global_step = None
        self.is_training = None
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
        # `is_training` not needed here but kept for API uniformity
        self.is_training = tf.placeholder(tf.bool, name='is_training')

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
            self.targets = targets

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
    def __init__(self, use_batch_norm=False):
        self.loss = None
        self.logits = None
        self.global_step = None
        self.use_batch_norm = use_batch_norm
        self.is_training = None
        self.reg = tf.contrib.layers.l2_regularizer(scale=0.0001)
        self.dropout_keep_prob = None
        self.padding = 'SAME'
        # note, 'NCHW' is only supported on GPUs
        self.data_format = 'NHWC'
        self.n_classes = 10
        self.layer_creator = LayerCreator('l2', 0.0001)

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
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        lc = self.layer_creator

        def make_wbkernels(
                name, shp, initializer=xavier_init(uniform=False),
        ):
            """ make weights, biases, kernels """
            return tf.get_variable(
                name, shp, initializer=initializer, regularizer=self.reg
            )

        def make_fc_layer(
                inp_lyr, name_w, shp_w, name_b=None, shp_b=None
        ):
            """ assume if shp_b is None that we are using batch norm. """
            if shp_b is None and self.use_batch_norm:
                W = lc.make_wbkernels(name_w, shp_w)
                fc_lyr = tf.matmul(inp_lyr, W, name='matmul')
                fc_lyr = tf.contrib.layers.batch_norm(
                    fc_lyr, center=True, scale=True,
                    data_format=self.data_format, is_training=self.is_training
                )
            else:
                W = make_wbkernels(name_w, shp_w)
                b = make_wbkernels(name_b, shp_b)
                fc_lyr = tf.nn.bias_add(
                    tf.matmul(inp_lyr, W, name='matmul'), b,
                    data_format=self.data_format, name='bias_add'
                )
            return fc_lyr
            
        def make_active_fc_Layer(
                inp_lyr, name_fc_lyr,
                name_w, shp_w, name_b=None, shp_b=None, act=tf.nn.relu
        ):
            fc_lyr = make_fc_layer(inp_lyr, name_w, shp_w, name_b, shp_b)
            fc_lyr = act(fc_lyr, name=name_fc_lyr)
            return fc_lyr

        def make_active_conv(
                input_lyr, name, kernels, biases=None, act=tf.nn.relu
        ):
            conv = tf.nn.conv2d(
                input_lyr, kernels, strides=[1, 1, 1, 1],
                padding=self.padding, data_format=self.data_format
            )
            if biases is None and self.use_batch_norm:
                # TODO - test `activation_fun` argument
                return act(tf.contrib.layers.batch_norm(
                    conv, center=True, scale=True,
                    data_format=self.data_format, is_training=self.is_training
                ))
            else:
                return act(tf.nn.bias_add(
                    conv, biases, data_format=self.data_format, name=name
                ))

        def make_pool(input_lyr, name):
            if self.data_format == 'NHWC':
                ksize = [1, 2, 2, 1]
                strides = [1, 2, 2, 1]
            elif self.data_format == 'NCHW':
                ksize = [1, 1, 2, 2]
                strides = [1, 1, 2, 2]
            else:
                raise ValueError('Invalid data format!')
            pool = tf.nn.max_pool(
                input_lyr, ksize=ksize, strides=strides,
                padding=self.padding, data_format=self.data_format,
                name=name
            )
            return pool

        with tf.variable_scope('io'):
            self.X_img = features

        with tf.variable_scope('model'):
            with tf.variable_scope('conv1'):
                self.kernels1 = make_wbkernels('kernels', [5, 5, 1, 32])
                self.biases1 = None if self.use_batch_norm else \
                    make_wbkernels('biases', [32])
                self.conv1 = make_active_conv(
                    self.X_img, 'relu_conv1', self.kernels1, self.biases1,
                )

            with tf.variable_scope('pool1'):
                self.pool1 = make_pool(self.conv1, name='pool1')

            with tf.variable_scope('conv2'):
                self.kernels2 = make_wbkernels('kernels', [5, 5, 32, 64])
                self.biases2 = None if self.use_batch_norm else \
                    make_wbkernels('biases', [64])
                self.conv2 = make_active_conv(
                    self.pool1, 'relu_conv2', self.kernels2, self.biases2,
                )

            with tf.variable_scope('pool2'):
                self.pool2 = make_pool(self.conv2, name='pool2')

            with tf.variable_scope('fc'):
                # use weight of dimension 7 * 7 * 64 x 1024
                input_features = 7 * 7 * 64
                # reshape pool2 to 2 dimensional
                self.pool2 = tf.reshape(self.pool2, [-1, input_features])
                self.fc = make_active_fc_Layer(
                    self.pool2, 'fully_connected',
                    'weights', [input_features, 1024], 'biases', [1024]
                )
                self.fc = tf.nn.dropout(
                    self.fc, self.dropout_keep_prob, name='relu_dropout'
                )

            with tf.variable_scope('softmax_linear'):
                self.weights_softmax = make_wbkernels(
                    'weights', [1024, self.n_classes]
                )
                self.biases_softmax = make_wbkernels(
                    'biases', [self.n_classes]
                )
                self.logits = tf.nn.bias_add(
                    tf.matmul(self.fc, self.weights_softmax),
                    self.biases_softmax, data_format=self.data_format,
                    name='logits'
                )

    def _set_targets(self, targets):
        with tf.variable_scope('targets'):
            self.targets = targets

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
        return ['model/softmax_linear/logits']
