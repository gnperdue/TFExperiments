from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import h5py
import numpy as np
import tensorflow as tf

# Get path to data
# find HDF5 here: wget https://raw.githubusercontent.com/gnperdue/RandomData/master/hdf5/fashion_test.hdf5
TFILE = os.path.join(
    os.environ['HOME'], 'Dropbox/Data/RandomData/hdf5/fashion_test.hdf5'
)


class FashionHDF5Reader(object):

    def __init__(self, hdf5_file):
        self._file = hdf5_file
        self._f = None
        self._nlabels = 10
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]

    def openf(self):
        self._f = h5py.File(self._file, 'r')
        self._nevents = self._f['fashion/labels'].shape[0]
        return self._nevents

    def closef(self):
        try:
            self._f.close()
        except AttributeError:
            print('hdf5 file is not open yet.')

    def get_examples(self, start_idx, stop_idx):
        image = self._f['fashion/images'][start_idx: stop_idx]
        image = np.moveaxis(image, 1, -1)
        label = self._f['fashion/labels'][start_idx: stop_idx].reshape([-1])
        oh_label = np.zeros((label.size, self._nlabels), dtype=np.uint8)
        oh_label[np.arange(label.size), label] = 1
        return image, oh_label


def _make_fashion_generator_fn(file_name, batch_size):
    """
    make a generator function that we can query for batches
    """
    reader = FashionHDF5Reader(file_name)
    nevents = reader.openf()

    def example_generator_fn():
        start_idx, stop_idx = 0, batch_size
        while True:
            if start_idx >= nevents:
                reader.closef()
                return
            yield reader.get_examples(start_idx, stop_idx)
            start_idx, stop_idx = start_idx + batch_size, stop_idx + batch_size

    return example_generator_fn


def make_fashion_dset(file_name, batch_size, shuffle=False):
    dgen = _make_fashion_generator_fn(file_name, batch_size)
    features_shape = [None, 28, 28, 1]
    labels_shape = [None, 10]
    ds = tf.data.Dataset.from_generator(
        dgen, (tf.float32, tf.uint8),
        (tf.TensorShape(features_shape), tf.TensorShape(labels_shape))
    )
    # we are grabbing an entire "batch", so don't call `batch()`, etc.
    ds = ds.prefetch(10)
    if shuffle:
        ds = ds.shuffle(10)

    return ds


def make_fashion_iterators(file_name, batch_size, shuffle=False):
    ds = make_fashion_dset(file_name, batch_size, shuffle)
    itrtr = ds.make_one_shot_iterator()
    images, labels = itrtr.get_next()
    return images, labels


images, labels = make_fashion_iterators(TFILE, 11)
with tf.Session() as sess:
    total_batches = 0
    total_examples = 0
    try:
        while True:
            im, ls = sess.run([images, labels])
            print('{}, {}, {}, {}'.format(
                im.shape, im.dtype, ls.shape, ls.dtype
            ))
            total_batches += 1
            total_examples += ls.shape[0]
    except tf.errors.OutOfRangeError:
        print('end of dataset at total_batches={}'.format(
            total_batches
        ))
    except Exception as e:
        print(e)
