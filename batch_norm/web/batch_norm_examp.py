"""
following http://ruishu.io/2016/12/27/batchnorm/
"""
from six.moves import range
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


def dense(x, size, scope):
    return tf.contrib.layers.fully_connected(
        x, size, activation_fn=None, scope=scope
    )


def dense_relu(x, size, scope):
    with tf.variable_scope(scope):
        h1 = dense(x, size, 'dense')
        return tf.nn.relu(h1, 'relu')


def dense_batch_relu(x, phase, scope):
    """
    check tf.contrib.layers.batch_norm documentation:
    NOTE: when training, the `moving_mean` and `moving_variance` _must_ be
    updated; by default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`,
    so they must be _added_ as a dependency to the `train_op`.

    You may set the `updates_colllections=None` to force updates in place, but
    this will impose a speed penalty.
    """
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(
            x, 100, activation_fn=None, scope='dense'
        )
        h2 = tf.contrib.layers.batch_norm(
            h1, center=True, scale=True, is_training=phase, scope='bn'
        )
        return tf.nn.relu(h2, 'relu')
    

tf.reset_default_graph()
x = tf.placeholder('float32', (None, 784), name='x')
y = tf.placeholder('float32', (None, 10), name='y')
phase = tf.placeholder(tf.bool, name='phase')

h1 = dense_batch_relu(x, phase, 'layer1')
h2 = dense_batch_relu(h1, phase, 'layer2')
logits = dense(h2, 10, scope='logits')

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1)), 'float32'
        )
    )
    
with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    )
    

def train():
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    history = []
    iterep = 500
    for i in range(iterep * 30):
        x_train, y_train = mnist.train.next_batch(100)
        sess.run(train_step,
                 feed_dict={'x:0': x_train,
                            'y:0': y_train,
                            'phase:0': 1})
        if (i + 1) % iterep == 0:
            epoch = (i + 1) / iterep
            tr = sess.run([loss, accuracy],
                          feed_dict={'x:0': mnist.train.images,
                                     'y:0': mnist.train.labels,
                                     'phase:0': 1})
            t = sess.run([loss, accuracy],
                         feed_dict={'x:0': mnist.test.images,
                                    'y:0': mnist.test.labels,
                                    'phase:0': 0})
            history += [[epoch] + tr + t]
            print history[-1]

    return history


history = train()
