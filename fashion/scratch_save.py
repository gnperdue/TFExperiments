import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
from tffashion.hdf5_readers import FashionHDF5Reader


data_dir = "/Users/perdue/Dropbox/Data/RandomData/hdf5"
TESTFILE = os.path.join(data_dir, 'fashion_test.hdf5')
TRAINFILE = os.path.join(data_dir, 'fashion_train.hdf5')

train_data = FashionHDF5Reader(TRAINFILE, tofloat=True)
nevents = train_data.openf()
print(nevents)


imgs, labels = train_data.get_examples(0, 20)
print(imgs.shape, imgs.dtype, labels.shape)
n_features = np.prod(imgs[0].shape)

tf.reset_default_graph()

X_tensor = tf.placeholder(tf.float32, [None, 28, 28, 1])

n_filters = [16, 16, 16]
filter_sizes = [4, 4, 4]

current_input = X_tensor
n_input = 1   # greyscale -> 1 channel
Ws = []
shapes = []

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
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))

        h = tf.nn.conv2d(current_input, W,
                         strides=[1, 2, 2, 1], padding='SAME')
        current_input = tf.nn.relu(h)

        Ws.append(W)
        n_input = n_output

# encoded = current_input

Ws.reverse()
shapes.reverse()
n_filters.reverse()
n_filters = n_filters[1:] + [1]

print(n_filters, filter_sizes, shapes)


for layer_i, shape in enumerate(shapes):
    with tf.variable_scope("decoder/layer/{}".format(layer_i)):
        W = Ws[layer_i]
        h = tf.nn.conv2d_transpose(
            current_input, W,
            tf.stack([tf.shape(X_tensor)[0], shape[1], shape[2], shape[3]]),
            strides=[1, 2, 2, 1], padding='SAME'
        )
        current_input = tf.nn.relu(h)

Y = current_input
cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(X_tensor, Y), 1))
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 100
n_epochs = 5
batches = [(i * batch_size, (i + 1) * batch_size)
           for i in range(nevents // batch_size)]

for epoch_i in range(n_epochs):
    for batch in batches:
        batch_X, _ = train_data.get_examples(batch[0], batch[1])
        sess.run(optimizer, feed_dict={X_tensor: batch_X})

recon = sess.run(Y, feed_dict={X_tensor: imgs})

print(recon.shape)
fig = plt.figure()
gs = plt.GridSpec(1, 2)
ax1 = plt.subplot(gs[0])
ax1.imshow(recon[0].reshape(28, 28))
ax2 = plt.subplot(gs[1])
ax2.imshow(imgs[0].reshape(28, 28))
plt.show()
