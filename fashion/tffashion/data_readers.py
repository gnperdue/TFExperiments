from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf


def _make_fashion_generator_fn(hdf5_file_name, use_oned_data=False):
    """
    make a generator function that we can query for batches
    """
    from tffashion.hdf5_readers import FashionHDF5Reader as HDF5Reader
    reader = HDF5Reader(hdf5_file_name)
    nevents = reader.openf()
    if use_oned_data:
        readfn = reader.get_flat_example
    else:
        readfn = reader.get_example

    def example_generator_fn():
        idx = 0
        while True:
            if idx >= nevents:
                reader.closef()
                return
            yield readfn(idx)
            idx += 1

    return example_generator_fn


def make_fashion_dset(hdf5_file, batch_size, num_epochs, use_oned_data=False):
    # make a generator function
    dgen = _make_fashion_generator_fn(hdf5_file, use_oned_data)

    # make a Dataset from a generator
    if use_oned_data:
        features_shape = [784]
    else:
        features_shape = [28, 28, 1]
    labels_shape = [10]
    ds = tf.data.Dataset.from_generator(
        dgen, (tf.float32, tf.uint8),
        (tf.TensorShape(features_shape), tf.TensorShape(labels_shape))
    )
    ds = ds.shuffle(1000).batch(batch_size).repeat(num_epochs)
    return ds


def make_fashion_iterators(
        hdf5_file, batch_size, num_epochs, use_oned_data=False
):
    '''
    estimators require an input fn returning `(features, labels)` pairs, where
    `features` is a dictionary of features.
    '''
    ds = make_fashion_dset(hdf5_file, batch_size, num_epochs, use_oned_data)

    # one_shot_iterators do not have initializers
    itrtr = ds.make_one_shot_iterator()
    feats, labs = itrtr.get_next()
    return feats, labs
    # return ds


def make_train_input_function(shuffle=True, shuffle_depth=1000):
    '''
    estimators require an input fn with a specific signature - customize that
    function here.
    '''
    def train_input_fn(features, labels, batch_size):
        pass

    return train_input_fn
