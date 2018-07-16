import numpy as np
import tensorflow as tf

# text from 
#    tensorboard --logdir /tmp/logdir --debugger_port 7000
# -> generates instructons for included text
from tensorflow.python import debug as tf_debug

k_true = [[1, -1], [3, -3], [2, -2]]
b_true = [-5, 5]
num_examples = 128

with tf.Session() as sess:

    x = tf.placeholder(tf.float32, shape=[None, 3], name='x')
    y = tf.placeholder(tf.float32, shape=[None, 2], name='y')

    dense_layer = tf.keras.layers.Dense(2, use_bias=True)
    y_hat = dense_layer(x)
    loss = tf.reduce_mean(
        tf.keras.losses.mean_squared_error(y, y_hat),
        name='loss'
    )
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    sess.run(tf.global_variables_initializer())

    # generated instructons from tensorboard
    sess = tf_debug.TensorBoardDebugWrapperSession(sess, "mac-131269:7000")

    for i in range(50):
        xs = np.random.randn(num_examples, 3)
        ys = np.matmul(xs, k_true) + b_true

        loss_val, _ = sess.run([loss, train_op], feed_dict={x: xs, y: ys})
        print('Iteration {}, loss = {}'.format(i, loss_val))
