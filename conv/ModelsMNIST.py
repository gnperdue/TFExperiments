"""
MNIST models
"""
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier_init

import logging

LOGGER = logging.getLogger(__name__)


class LayerCreator:
    def __init__(
            self, regularization_strategy='l2', regularization_scale=0.0001,
            use_batch_norm=False, data_format='NHWC'
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
        self.use_batch_norm = use_batch_norm
        self.batch_norm_decay = 0.999
        self.is_training = None
        self.data_format = data_format
        self.padding = 'SAME'   # TODO must be more flexible
        # self.dropout_config??

    def set_is_training_placeholder(self, is_training):
        self.is_training = is_training

    def make_wbkernels(
            self, name, shp=None, initializer=xavier_init(uniform=False),
    ):
        """ make weights, biases, kernels """
        return tf.get_variable(
            name, shp, initializer=initializer, regularizer=self.reg
        )

    def make_fc_layer(
            self, inp_lyr, name_fc_lyr,
            name_w, shp_w, name_b=None, shp_b=None,
            initializer=xavier_init(uniform=False)
    ):
        """ TODO - regularize batch norm params? """
        W = self.make_wbkernels(name_w, shp_w, initializer=initializer)
        b = self.make_wbkernels(
            name_b, shp_b, initializer=tf.zeros_initializer()
        )
        fc_lyr = tf.nn.bias_add(
            tf.matmul(inp_lyr, W, name=name_fc_lyr+'_matmul'), b,
            data_format=self.data_format, name=name_fc_lyr,
        )
        if self.use_batch_norm:
            fc_lyr = tf.contrib.layers.batch_norm(
                fc_lyr, decay=self.batch_norm_decay, center=True, scale=True,
                data_format=self.data_format, is_training=self.is_training
            )
        return fc_lyr
            
    def make_active_fc_layer(
            self, inp_lyr, name_fc_lyr,
            name_w, shp_w, name_b=None, shp_b=None,
            act=tf.nn.relu, initializer=xavier_init(uniform=False)
    ):
        return act(self.make_fc_layer(
            inp_lyr, name_fc_lyr, name_w, shp_w, name_b, shp_b,
            initializer=initializer
        ), name=name_fc_lyr+'_act')

    def make_active_conv(
            self, input_lyr, name, kernels, biases=None, act=tf.nn.relu
    ):
        """ TODO - regularize batch norm params? biases? """
        conv = tf.nn.conv2d(
            input_lyr, kernels, strides=[1, 1, 1, 1],
            padding=self.padding, data_format=self.data_format,
            name=name
        )
        if biases is None and self.use_batch_norm:
            # TODO - test `activation_fun` argument
            return act(tf.contrib.layers.batch_norm(
                conv, decay=self.batch_norm_decay, center=True, scale=True,
                data_format=self.data_format, is_training=self.is_training
            ))
        else:
            return act(tf.nn.bias_add(
                conv, biases, data_format=self.data_format, name=name+'_act'
            ))

    def make_pool(
            self, input_lyr, name, h_step=2, w_step=2, n_step=1, c_step=1,
            padding=None
    ):
        padding = padding or self.padding
        if self.data_format == 'NHWC':
            ksize = [n_step, h_step, w_step, c_step]
            strides = [n_step, h_step, w_step, c_step]
        elif self.data_format == 'NCHW':
            ksize = [n_step, c_step, h_step, w_step]
            strides = [n_step, c_step, h_step, w_step]
        else:
            raise ValueError('Invalid data format!')
        pool = tf.nn.max_pool(
            input_lyr, ksize=ksize, strides=strides,
            padding=padding, data_format=self.data_format, name=name
        )
        return pool


def create_train_valid_summaries(
        loss, accuracy=None, reg_loss=None, do_valid=True
):
    base_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    train_summaries, valid_summaries = [], []
    train_summary_op, valid_summary_op = None, None
    with tf.name_scope('summaries/train'):
        train_loss = tf.summary.scalar('loss', loss)
        train_histo_loss = tf.summary.histogram(
            'histogram_loss', loss
        )
        train_summaries.extend([train_loss, train_histo_loss])
        if reg_loss is not None:
            train_reg_loss = tf.summary.scalar('reg_loss', reg_loss)
            train_summaries.append(train_reg_loss)
        if accuracy is not None:
            train_accuracy = tf.summary.scalar('accuracy', accuracy)
            train_summaries.append(train_accuracy)
        train_summaries.extend(base_summaries)
        train_summary_op = tf.summary.merge(train_summaries)
    if do_valid:
        with tf.name_scope('summaries/valid'):
            valid_loss = tf.summary.scalar('loss', loss)
            valid_histo_loss = tf.summary.histogram(
                'histogram_loss', loss
            )
            valid_summaries.extend([valid_loss, valid_histo_loss])
            if reg_loss is not None:
                valid_reg_loss = tf.summary.scalar('reg_loss', reg_loss)
                valid_summaries.append(valid_reg_loss)
            if accuracy is not None:
                valid_accuracy = tf.summary.scalar('accuracy', accuracy)
                valid_summaries.append(valid_accuracy)
            valid_summaries.extend(base_summaries)
            valid_summary_op = tf.summary.merge(valid_summaries)
    return train_summary_op, valid_summary_op


def compute_categorical_loss_and_accuracy(logits, targets):
    """return total loss, reg loss (subset of total), and accuracy"""
    with tf.variable_scope('loss'):
        regularization_losses = sum(
            tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES
            )
        )
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=targets
            ),
            axis=0,
            name='loss'
        ) + regularization_losses
        preds = tf.nn.softmax(logits, name='preds')
        correct_preds = tf.equal(
            tf.argmax(preds, 1), tf.argmax(targets, 1),
            name='correct_preds'
        )
        accuracy = tf.divide(
            tf.reduce_sum(tf.cast(correct_preds, tf.float32)),
            tf.cast(tf.shape(targets)[0], tf.float32),
            name='accuracy'
        )
    return loss, regularization_losses, accuracy


def make_standard_placeholders():
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
    global_step = tf.Variable(
        0, dtype=tf.int32, trainable=False, name='global_step'
    )
    is_training = tf.placeholder(tf.bool, name='is_training')
    return dropout_keep_prob, global_step, is_training


class MNISTLogReg:
    def __init__(self):
        self.loss = None
        self.logits = None
        self.global_step = None
        self.is_training = None
        self.reg = tf.contrib.layers.l2_regularizer(scale=0.0001)
        # dropout not used here, but kept for API uniformity
        self.dropout_keep_prob = None
        self.layer_creator = LayerCreator('l2', 0.0001)

    def _create_summaries(self):
        self.train_summary_op, self.valid_summary_op =  \
            create_train_valid_summaries(
                self.loss, self.accuracy, self.regularization_losses
            )

    def _build_network(self, features):

        self.dropout_keep_prob, self.global_step, self.is_training = \
            make_standard_placeholders()
        lc = self.layer_creator
        lc.set_is_training_placeholder(self.is_training)

        with tf.variable_scope('io'):
            self.X = features

        with tf.variable_scope('model'):
            W1 = lc.make_wbkernels(
                name='weights1',
                initializer=tf.random_normal(
                    shape=[784, 10], mean=0.0, stddev=0.01, dtype=tf.float32
                )
            )
            b1 = lc.make_wbkernels(
                name='bias1',
                initializer=tf.random_normal(
                    shape=[10], mean=0.0, stddev=0.01, dtype=tf.float32
                )
            )
            self.logits = tf.add(
                tf.matmul(self.X, W1), b1, name='logits'
            )

    def _set_targets(self, targets):
        with tf.variable_scope('targets'):
            self.targets = targets

    def _define_loss(self):
        self.loss, self.regularization_losses, self.accuracy = \
            compute_categorical_loss_and_accuracy(self.logits, self.targets)

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


class MNISTMLP:
    def __init__(self, use_batch_norm=False, use_dropout=False):
        self.loss = None
        self.logits = None
        self.global_step = None
        self.is_training = None
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_keep_prob = None
        self.padding = 'SAME'
        self.data_format = 'NHWC'
        self.layer_creator = LayerCreator(
            'l2', 0.0, self.use_batch_norm, self.data_format
        )
        self.n_classes = 10

    def _create_summaries(self):
        self.train_summary_op, self.valid_summary_op =  \
            create_train_valid_summaries(
                self.loss, self.accuracy, self.regularization_losses
            )

    def _build_network(self, features):

        self.dropout_keep_prob, self.global_step, self.is_training = \
            make_standard_placeholders()
        lc = self.layer_creator
        lc.set_is_training_placeholder(self.is_training)

        with tf.variable_scope('io'):
            self.X = features

        with tf.variable_scope('model'):
            with tf.variable_scope('fc_lyr1'):
                fc_lyr1 = lc.make_active_fc_layer(
                    self.X, 'fully_connected',
                    'weights', [784, 100],
                    'biases', [100],
                    initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=0.01
                    )
                )
                if self.use_dropout:
                    fc_lyr1 = tf.nn.dropout(
                        fc_lyr1, self.dropout_keep_prob, name='relu_dropout'
                    )
            with tf.variable_scope('fc_lyr2'):
                fc_lyr2 = lc.make_active_fc_layer(
                    fc_lyr1, 'fully_connected',
                    'weights', [100, 100],
                    'biases', [100],
                    initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=0.01
                    )
                )
                if self.use_dropout:
                    fc_lyr2 = tf.nn.dropout(
                        fc_lyr2, self.dropout_keep_prob, name='relu_dropout'
                    )
            with tf.variable_scope('final_linear'):
                W = lc.make_wbkernels(
                    'weights', [100, self.n_classes],
                    initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=0.01
                    )
                )
                b = lc.make_wbkernels(
                    'biases', [self.n_classes],
                    initializer=tf.zeros_initializer()
                )
                self.logits = tf.nn.bias_add(
                    tf.matmul(fc_lyr2, W, name='matmul'), b,
                    data_format=self.data_format, name='logits',
                )

    def _set_targets(self, targets):
        with tf.variable_scope('targets'):
            self.targets = targets

    def _define_loss(self):
        self.loss, self.regularization_losses, self.accuracy = \
            compute_categorical_loss_and_accuracy(self.logits, self.targets)

    def _define_train_op(self, learning_rate):
        LOGGER.info('Building train op with learning_rate = %f' %
                    learning_rate)
        with tf.variable_scope('training'):
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
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
        return ['model/final_linear/logits']


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
        self.data_format = 'NHWC'
        self.n_classes = 10
        self.layer_creator = LayerCreator(
            'l2', 0.0001, use_batch_norm, self.data_format
        )

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

        self.dropout_keep_prob, self.global_step, self.is_training = \
            make_standard_placeholders()
        lc = self.layer_creator
        lc.set_is_training_placeholder(self.is_training)

        with tf.variable_scope('io'):
            self.X_img = features

        with tf.variable_scope('model'):
            with tf.variable_scope('conv1'):
                self.kernels1 = lc.make_wbkernels('kernels', [5, 5, 1, 32])
                self.biases1 = None if self.use_batch_norm else \
                    lc.make_wbkernels('biases', [32])
                self.conv1 = lc.make_active_conv(
                    self.X_img, 'relu_conv1', self.kernels1, self.biases1,
                )

            with tf.variable_scope('pool1'):
                self.pool1 = lc.make_pool(self.conv1, name='pool1')

            with tf.variable_scope('conv2'):
                self.kernels2 = lc.make_wbkernels('kernels', [5, 5, 32, 64])
                self.biases2 = None if self.use_batch_norm else \
                    lc.make_wbkernels('biases', [64])
                self.conv2 = lc.make_active_conv(
                    self.pool1, 'relu_conv2', self.kernels2, self.biases2,
                )

            with tf.variable_scope('pool2'):
                self.pool2 = lc.make_pool(self.conv2, name='pool2')

            with tf.variable_scope('fc'):
                # use weight of dimension 7 * 7 * 64 x 1024
                input_features = 7 * 7 * 64
                # reshape pool2 to 2 dimensional
                self.pool2 = tf.reshape(self.pool2, [-1, input_features])
                biases_name = None if self.use_batch_norm else 'biases'
                biases_shape = None if self.use_batch_norm else [1024]
                self.fc = lc.make_active_fc_layer(
                    self.pool2, 'fully_connected',
                    'weights', [input_features, 1024],
                    biases_name, biases_shape
                )
                self.fc = tf.nn.dropout(
                    self.fc, self.dropout_keep_prob, name='relu_dropout'
                )

            with tf.variable_scope('softmax_linear'):
                self.weights_softmax = lc.make_wbkernels(
                    'weights', [1024, self.n_classes]
                )
                self.biases_softmax = lc.make_wbkernels(
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
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
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
