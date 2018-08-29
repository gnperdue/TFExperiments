from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import tensorflow as tf


def _make_fashion_generator_fn(file_name):
    """
    make a generator function that we can query for batches
    """
    from tffashion.hdf5_readers import FashionHDF5Reader as HDF5Reader
    reader = HDF5Reader(file_name)
    nevents = reader.openf()

    def example_generator_fn():
        idx = 0
        while True:
            if idx >= nevents:
                reader.closef()
                return
            yield reader.get_example(idx)
            idx += 1

    return example_generator_fn


def _parse_mnist_tfrec(tfrecord):
    tfrecord_features = tf.parse_single_example(
        tfrecord,
        features={
            'images': tf.FixedLenFeature([], tf.string),
            'labels': tf.FixedLenFeature([], tf.string)
        },
        name='data'
    )
    images = tf.decode_raw(tfrecord_features['images'], tf.uint8)
    # note, 'NCHW' is only supported on GPUs, so use 'NHWC'...
    images = tf.reshape(images, [-1, 28, 28, 1])
    images = tf.cast(images, tf.float32)
    labels = tf.decode_raw(tfrecord_features['labels'], tf.uint8)
    labels = tf.one_hot(indices=labels, depth=10, on_value=1, off_value=0)
    labels = tf.reshape(labels, [10])
    return images, labels


def make_fashion_dset(
    file_name, batch_size, num_epochs=1, tfrecord=False
):
    if tfrecord:
        ds = tf.data.TFRecordDataset([file_name], compression_type='GZIP')
        # if shuffle:
        #     ds = ds.shuffle(buffer_size=256)
        ds = ds.map(_parse_mnist_tfrec).prefetch(batch_size)
        ds = ds.batch(batch_size).repeat(num_epochs)
    else:
        # make a generator function - read from HDF5
        dgen = _make_fashion_generator_fn(file_name)

        # make a Dataset from a generator
        features_shape = [28, 28, 1]
        labels_shape = [10]
        ds = tf.data.Dataset.from_generator(
            dgen, (tf.float32, tf.uint8),
            (tf.TensorShape(features_shape), tf.TensorShape(labels_shape))
        )
        ds = ds.prefetch(10*batch_size)
        ds = ds.shuffle(10*batch_size).batch(batch_size).repeat(num_epochs)

    return ds


def make_fashion_iterators(
        file_name, batch_size, num_epochs=1, tfrecord=False
):
    '''
    estimators require an input fn returning `(features, labels)` pairs, where
    `features` is a dictionary of features.

    TODO - pass a shuffle flag
    '''
    ds = make_fashion_dset(file_name, batch_size, num_epochs, tfrecord)

    # one_shot_iterators do not have initializers
    itrtr = ds.make_one_shot_iterator()
    feats, labs = itrtr.get_next()
    return feats, labs


def get_data_files_dict(path='path_to_data', tfrecord=False):
    data_dict = {}
    if tfrecord:
        data_dict['train'] = os.path.join(path, 'fashion_train.tfrecord.gz')
        data_dict['test'] = os.path.join(path, 'fashion_test.tfrecord.gz')
    else:
        data_dict['train'] = os.path.join(path, 'fashion_train.hdf5')
        data_dict['test'] = os.path.join(path, 'fashion_test.hdf5')
    return data_dict