"""
MNIST models
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf


def two_layer_conv(features, params_dict):
    lc = params_dict['layer_creator']
    dropout_keep_prob = params_dict['dropout_keep_prob']
    n_classes = params_dict['n_classes']
    use_dropout = params_dict['use_dropout']
    data_format = params_dict['data_format']

    with tf.variable_scope('model'):
        with tf.variable_scope('conv1'):
            kernels1 = lc.make_wbkernels('kernels', [5, 5, 1, 32])
            biases1 = lc.make_wbkernels(
                'biases', [32], initializer=tf.zeros_initializer()
            )
            conv1 = lc.make_active_conv(
                features, 'relu_conv1', kernels1, biases1,
            )

        with tf.variable_scope('pool1'):
            pool1 = lc.make_pool(conv1, name='pool1')

        with tf.variable_scope('conv2'):
            kernels2 = lc.make_wbkernels('kernels', [5, 5, 32, 64])
            biases2 = lc.make_wbkernels(
                'biases', [64], initializer=tf.zeros_initializer()
            )
            conv2 = lc.make_active_conv(
                pool1, 'relu_conv2', kernels2, biases2,
            )

        with tf.variable_scope('pool2'):
            pool2 = lc.make_pool(conv2, name='pool2')

        with tf.variable_scope('fc'):
            # use weight of dimension 7 * 7 * 64 x 1024
            input_features = 7 * 7 * 64
            # reshape pool2 to 2 dimensional
            pool2 = tf.reshape(pool2, [-1, input_features])
            fc = lc.make_active_fc_layer(
                pool2, 'fully_connected',
                'weights', [input_features, 1024],
                'biases', [1024],
                initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01
                )
            )
            if use_dropout:
                fc = tf.nn.dropout(
                    fc, dropout_keep_prob, name='relu_dropout'
                )

        with tf.variable_scope('softmax_linear'):
            weights_softmax = lc.make_wbkernels('weights', [1024, n_classes])
            biases_softmax = lc.make_wbkernels('biases', [n_classes])
            logits = tf.nn.bias_add(
                tf.matmul(fc, weights_softmax),
                biases_softmax, data_format=data_format,
                name='logits'
            )

    return logits


def two_layer_mlp(features, params_dict):
    lc = params_dict['layer_creator']
    dropout_keep_prob = params_dict['dropout_keep_prob']
    n_classes = params_dict['n_classes']
    use_dropout = params_dict['use_dropout']
    data_format = params_dict['data_format']

    with tf.variable_scope('model'):
        with tf.variable_scope('fc_lyr1'):
            fc_lyr1 = lc.make_active_fc_layer(
                features, 'fully_connected',
                'weights', [784, 100],
                'biases', [100],
                initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01
                )
            )
            if use_dropout:
                fc_lyr1 = tf.nn.dropout(
                    fc_lyr1, dropout_keep_prob, name='relu_dropout'
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
            if use_dropout:
                fc_lyr2 = tf.nn.dropout(
                    fc_lyr2, dropout_keep_prob, name='relu_dropout'
                )
        with tf.variable_scope('final_linear'):
            W = lc.make_wbkernels(
                'weights', [100, n_classes],
                initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01
                )
            )
            b = lc.make_wbkernels(
                'biases', [n_classes],
                initializer=tf.zeros_initializer()
            )
            logits = tf.nn.bias_add(
                tf.matmul(fc_lyr2, W, name='matmul'), b,
                data_format=data_format, name='logits',
            )

    return logits
