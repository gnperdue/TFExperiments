'''
'bare' TensorFlow models - use the 'old-school' APIs here in a lot of places,
so don't expect good behavior with eager execution, etc.
'''
import numpy as np
import tensorflow as tf


class FashionMNISTLogReg:
    '''simple logistic regression'''

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


class FashionAutoencoder:
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

    def build_network(self, features):

        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step'
        )

        with tf.variable_scope('io'):
            self.X = features

        with tf.variable_scope('model'):
            n_filters = [16, 16, 16]
            filter_sizes = [4, 4, 4]

            current_input = self.X
            n_input = 1   # greyscale -> 1 channel
            Ws = []
            shapes = []
            final_encoder_layer = None

            for layer_i, n_output in enumerate(n_filters):
                with tf.variable_scope("encoder/layer/{}".format(layer_i)):
                    shapes.append(current_input.get_shape().as_list())
                    W = tf.get_variable(
                        name='W',
                        shape=[
                            filter_sizes[layer_i],
                            filter_sizes[layer_i],
                            n_input,
                            n_output],
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=0.02
                        )
                    )
                    h = tf.nn.conv2d(current_input, W,
                                     strides=[1, 1, 1, 1], padding='SAME')
                    current_input = tf.nn.relu(h)

                    Ws.append(W)
                    n_input = n_output
                    final_encoder_layer = layer_i

            # need to flatten, compress further
            with tf.variable_scope("encoder"):
                final_conv = current_input
                final_conv_shp = final_conv.get_shape().as_list()
                nfeat_flattened = np.prod(final_conv_shp[1:])
                current_input = tf.reshape(final_conv, [-1, nfeat_flattened])
                n_input = nfeat_flattened
                final_encoder_layer += 1

            dimensions = [128, 64]
            flat_Ws = []
            for layer_i, n_output in enumerate(dimensions):
                layer_n = layer_i + final_encoder_layer
                with tf.variable_scope("encoder/layer/{}".format(layer_n)):
                    W = tf.get_variable(
                        name='W',
                        shape=[n_input, n_output],
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=0.02
                        )
                    )
                    h = tf.matmul(current_input, W)
                    current_input = tf.nn.relu(h)
                    flat_Ws.append(W)
                    n_input = n_output

            self.encoded = current_input

            flat_Ws = flat_Ws[::-1]
            dimensions = dimensions[::-1][1:] + [nfeat_flattened]
            final_decoder_layer = None

            for layer_i, n_output in enumerate(dimensions):
                with tf.variable_scope("decoder/layer/{}".format(layer_i)):
                    W = tf.transpose(flat_Ws[layer_i])
                    h = tf.matmul(current_input, W)
                    current_input = tf.nn.relu(h)
                    n_input = n_output
                    final_decoder_layer = layer_i

            with tf.variable_scope("decoder"):
                new_shape = [-1] + final_conv_shp[1:]
                current_input = tf.reshape(current_input, new_shape)

            Ws.reverse()
            shapes.reverse()
            n_filters.reverse()
            n_filters = n_filters[1:] + [1]
            final_decoder_layer += 1

            for layer_i, shape in enumerate(shapes):
                layer_n = layer_i + final_decoder_layer
                with tf.variable_scope("decoder/layer/{}".format(layer_n)):
                    W = Ws[layer_i]
                    h = tf.nn.conv2d_transpose(
                        current_input, W,
                        tf.stack([
                            tf.shape(self.X)[0], shape[1], shape[2], shape[3]
                        ]),
                        strides=[1, 1, 1, 1], padding='SAME'
                    )
                    current_input = tf.nn.relu(h)

            self.Y = current_input

        with tf.variable_scope('loss'):
            self.regularization_losses = sum(
                tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES
                )
            )
            self.loss = tf.reduce_mean(
                tf.reduce_mean(tf.squared_difference(self.X, self.Y), 1),
                name='loss'
            ) + self.regularization_losses

        with tf.variable_scope('training'):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate
            ).minimize(self.loss, global_step=self.global_step)

        self._create_summaries()
