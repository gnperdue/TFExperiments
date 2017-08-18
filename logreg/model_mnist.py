import tensorflow as tf


class MNISTLogReg:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.loss = None
        self.logits = None
        self.global_step = None

    def _create_summaries(self):
        with tf.name_scope('summaries/train'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            self.train_summary_op = tf.summary.merge_all()
        with tf.name_scope('summaries/valid'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            self.valid_summary_op = tf.summary.merge_all()

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
                )
            )
            self.b = tf.get_variable(
                name='bias',
                initializer=tf.random_normal(
                    shape=[10], mean=0.0, stddev=0.01, dtype=tf.float32
                )
            )
            self.logits = tf.add(
                tf.matmul(self.X, self.W), self.b, name='logits'
            )

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.Y
                ),
                axis=0,
                name='loss'
            )

        with tf.variable_scope('training'):
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.learning_rate
            ).minimize(self.loss, global_step=self.global_step)

        self._create_summaries()
