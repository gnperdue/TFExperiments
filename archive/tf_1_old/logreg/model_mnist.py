import tensorflow as tf


class MNISTLogReg:
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
