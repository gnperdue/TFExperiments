import tensorflow as tf
tfe = tf.contrib.eager


class MNISTLogReg:
    def __init__(self):
        self.reg = tf.contrib.layers.l2_regularizer(scale=0.0001)

    def _logits(self, features):
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

    def loss(self, features, labels):
        logits = self._logits(features)

        with tf.variable_scope('loss'):
            # this won't work in eager mode - no GraphKeys
            # regularization_losses = sum(
            #     tf.get_collection(
            #         tf.GraphKeys.REGULARIZATION_LOSSES
            #     )
            # )
            # regularization_losses = tf.losses.get_regularization_losses()
            regularization_losses = 0.0
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=labels
                ),
                axis=0,
                name='loss'
            ) + regularization_losses

        return loss

    def get_softmax(self, features):
        logits = self._logits(features)
        return tf.nn.softmax(logits)

    def predict(self, features):
        softmax = self.get_softmax(features)
        return tf.argmax(softmax, 1)
