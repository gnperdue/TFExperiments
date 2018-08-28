from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier_init


class LayerCreator:
    def __init__(
            self, training=False,
            regularization_strategy='l2', regularization_scale=0.0001,
            use_batch_norm=False, data_format='NHWC', padding='SAME'
    ):
        if regularization_strategy == 'l2':
            self.reg = tf.contrib.layers.l2_regularizer(
                scale=regularization_scale
            )
        else:
            raise NotImplementedError(
                'Regularization strategy ' + regularization_strategy
                + ' is not implemented yet.'
            )
        self.use_batch_norm = use_batch_norm
        self.batch_norm_decay = 0.999
        self.is_training = training
        self.data_format = data_format
        self.padding = padding

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
            self, input_lyr, name, kernels,
            biases=None, act=tf.nn.relu, strides=[1, 1, 1, 1],
            padding=None
    ):
        """ TODO - regularize batch norm params? biases? """
        padding = padding or self.padding
        conv = tf.nn.conv2d(
            input_lyr, kernels, strides=strides,
            padding=padding, data_format=self.data_format,
            name=name
        )
        if self.use_batch_norm:
            # TODO - test `activation_fun` argument
            return act(
                tf.contrib.layers.batch_norm(
                    tf.nn.bias_add(
                        conv, biases, data_format=self.data_format,
                        name=name+'_plus_biases'
                    ), decay=self.batch_norm_decay,
                    center=True, scale=True,
                    data_format=self.data_format, is_training=self.is_training
                ),
                name=name+'_act'
            )
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
