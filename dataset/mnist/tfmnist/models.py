import tensorflow as tf
tfe = tf.contrib.eager


class MNISTLogReg:
    def __init__(self, l2_reg_scale=0.0001):
        self.reg = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)

    def logits(self, features):

        with tf.variable_scope('model', regularizer=self.reg):
            W = tfe.Variable(
                initial_value=tf.random_normal(
                    shape=[784, 10], mean=0.0, stddev=0.01, dtype=tf.float32
                ),
                name='weights',
            )
            b = tfe.Variable(
                initial_value=tf.random_normal(
                    shape=[10], mean=0.0, stddev=0.01, dtype=tf.float32
                ),
                name='bias',
            )
            logits = tf.add(
                tf.matmul(features, W), b, name='logits'
            )

        return logits

    def loss(self, logits, targets):

        with tf.variable_scope('loss'):
            # this messes with tfe.Variable somehow... (obvs 0 in eager mode,
            # but funky even in graph mode)
            # regularization_losses = sum(
            #     tf.get_collection(
            #         tf.GraphKeys.REGULARIZATION_LOSSES
            #     )
            # )
            regularization_losses = 0.0
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=targets
                ),
                axis=0,
                name='loss'
            ) + regularization_losses

        return loss

    def softmax_predictions(self, logits):

        with tf.variable_scope('predictions'):
            sftmx_predictions = tf.nn.softmax(logits, name='predictions')

        return sftmx_predictions

    def accuracy(self, softmax_predictions, targets):
        """ predictions are softmax vectors, targets are one-hot """

        with tf.variable_scope('accuracy'):
            correct_predictions = tf.equal(
                tf.argmax(softmax_predictions, 1), tf.argmax(targets, 1),
                name='correct_predictions'
            )
            accuracy = tf.divide(
                tf.reduce_sum(tf.cast(correct_predictions, tf.float32)),
                tf.cast(tf.shape(targets)[0], tf.float32),
                name='accuracy'
            )

        return accuracy


