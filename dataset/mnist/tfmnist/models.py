import tensorflow as tf


class MNISTLogReg:
    def __init__(self):
        self.reg = tf.contrib.layers.l2_regularizer(scale=0.0001)

    def loss(self, features, targets):

        with tf.variable_scope('model'):
            W = tf.get_variable(
                name='weights',
                initializer=tf.random_normal(
                    shape=[784, 10], mean=0.0, stddev=0.01, dtype=tf.float32
                ),
                regularizer=self.reg
            )
            b = tf.get_variable(
                name='bias',
                initializer=tf.random_normal(
                    shape=[10], mean=0.0, stddev=0.01, dtype=tf.float32
                ),
                regularizer=self.reg
            )
            logits = tf.add(
                tf.matmul(features, W), b, name='logits'
            )

        with tf.variable_scope('loss'):
            regularization_losses = sum(
                tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES
                )
            )
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=targets
                ),
                axis=0,
                name='loss'
            ) + regularization_losses

        return loss

