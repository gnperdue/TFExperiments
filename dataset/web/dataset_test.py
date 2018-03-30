from __future__ import print_function
import numpy as np
import tensorflow as tf
import h5py
from six.moves import range


def read_hdf5file(filename, data_format='NHWC'):
    _f = h5py.File(filename, 'r')
    # hdf5 img data has shape (N, C, H, W)
    features = _f['features'][:]
    if data_format == 'NHWC':
        features = np.squeeze(features)
        features = np.expand_dims(features, axis=3)
    targets = _f['targets'][:]
    _f.close()
    return features, targets


dataset = tf.data.Dataset.from_tensor_slices(['mnist_test.hdf5'])
dataset = dataset.map(
    lambda filename: tuple(tf.py_func(
        read_hdf5file, [filename], [tf.uint8, tf.uint8]
    ))
)
dataset = dataset.repeat(1)
iterator = dataset.make_one_shot_iterator()
batch_features, batch_targets = iterator.get_next()
    
with tf.Session() as sess:
    for _ in range(3):
        try:
            f, l = sess.run([batch_features, batch_targets])
            print(f.shape, l.shape)
        except tf.errors.OutOfRangeError as e:
            print('end of sequence')
        except Exception as e:
            print(e)
