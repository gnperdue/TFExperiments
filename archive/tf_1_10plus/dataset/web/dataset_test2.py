from __future__ import print_function
import numpy as np
import tensorflow as tf
import h5py
from six.moves import range


def read_hdf5file(filename, min_value, max_value, data_format='NHWC'):
    _f = h5py.File(filename, 'r')
    # hdf5 img data has shape (N, C, H, W)
    features = _f['features'][min_value: max_value]
    if data_format == 'NHWC':
        features = np.squeeze(features)
        features = np.expand_dims(features, axis=3)
    targets = _f['targets'][min_value: max_value]
    _f.close()
    return features, targets


minv = tf.placeholder(tf.int64, shape=[])
maxv = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.from_tensor_slices([
    'mnist_test.hdf5'
])
dataset = dataset.map(
    lambda filename: tuple(tf.py_func(
        read_hdf5file, [filename, minv, maxv], [tf.uint8, tf.uint8]
    ))
)
dataset = dataset.repeat(1)
iterator = dataset.make_initializable_iterator()
batch_features, batch_targets = iterator.get_next()
    
batch_size = 512

with tf.Session() as sess:
    for i in range(20):
        min_idx = i * batch_size
        max_idx = i * batch_size + batch_size
        print(min_idx, max_idx)
        sess.run(
            iterator.initializer,
            feed_dict={minv: min_idx, maxv: max_idx}
        )
        try:
            f, l = sess.run([batch_features, batch_targets])
            print(f.shape, l.shape)
        except tf.errors.OutOfRangeError as e:
            print('end of sequence')
        except Exception as e:
            print(e)
