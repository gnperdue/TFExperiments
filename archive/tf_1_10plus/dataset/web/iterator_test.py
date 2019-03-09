from __future__ import print_function
import numpy as np
import tensorflow as tf
import h5py
from six.moves import range


class DataReader:
    def __init__(self, filename, data_format='NHWC'):
        self.filename = filename
        self.data_format = data_format
        self._f = None
        self.data_dict = {}
        self._read_hdf5()

    def _read_hdf5(self):
        self._f = h5py.File(self.filename, 'r')
        # hdf5 img data has shape (N, C, H, W)
        features = self._f['features'][:]
        if self.data_format == 'NHWC':
            features = np.squeeze(features)
            features = np.expand_dims(features, axis=3)
        self.data_dict['features'] = features
        self.data_dict['targets'] = self._f['targets'][:]
        self._f.close()


reader = DataReader('mnist_train.hdf5')
features = reader.data_dict['features']
targets = reader.data_dict['targets']

features_placeholder = tf.placeholder(features.dtype, features.shape)
targets_placeholder = tf.placeholder(targets.dtype, targets.shape)

dataset = tf.data.Dataset.from_tensor_slices((
    features_placeholder, targets_placeholder
))
dataset = dataset.repeat(1)
dataset = dataset.batch(10)
iterator = dataset.make_initializable_iterator()

with tf.Session() as sess:
    sess.run(
        iterator.initializer,
        feed_dict={
            features_placeholder: features,
            targets_placeholder: targets
        }
    )
    batch_features, batch_targets = iterator.get_next()
    for _ in range(3):
        try:
            f, l = sess.run([batch_features, batch_targets])
            print(f.shape, l.shape)
        except tf.errors.OutOfRangeError as e:
            print('end of sequence')
        except Exception as e:
            print(e)
