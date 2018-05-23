from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf


def _make_mnist_generator_fn(hdf5_file_name, use_oned_data=False):
    """
    make a generator function that we can query for batches
    """
    from tfconv.hdf5_readers import MNISTHDF5Reader as HDF5Reader
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


def make_mnist_dset(hdf5_file, batch_size, num_epochs, use_oned_data=False):
    # make a generator function
    dgen = _make_mnist_generator_fn(hdf5_file, use_oned_data)

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


def make_mnist_iterators(
        hdf5_file, batch_size, num_epochs, use_oned_data=False
):
    ds = make_mnist_dset(hdf5_file, batch_size, num_epochs, use_oned_data)

    # one_shot_iterators do not have initializers
    itrtr = ds.make_one_shot_iterator()
    feats, labs = itrtr.get_next()
    return feats, labs
    # return ds


def test_graph_one_shot_iterator_read(hdf5_file, batch_size=25, num_epochs=1):
    feats, labs = make_mnist_iterators(hdf5_file, batch_size, num_epochs)
    with tf.Session() as sess:
        counter = 0
        try:
            while True:
                fs, ls = sess.run([feats, labs])
                print(fs.shape, fs.dtype, ls.shape, ls.dtype)
                counter += 1
                if counter > 1000:
                    break
        except tf.errors.OutOfRangeError:
            print('end of dataset at counter = {}'.format(counter))
        except Exception as e:
            print(e)


def test_eager_one_shot_iterator_read(hdf5_file, batch_size=25, num_epochs=1):
    tfe = tf.contrib.eager
    tf.enable_eager_execution()
    targets_and_labels = make_mnist_dset(
        hdf5_file, batch_size, num_epochs, use_oned_data=True
    )

    for i, (fs, ls) in enumerate(tfe.Iterator(targets_and_labels)):
        print(fs.shape, fs.dtype, ls.shape, ls.dtype)
        

if __name__ == '__main__':
    import os
    # Get path to data
    DDIR = os.environ['HOME'] + '/Dropbox/Data/RandomData/hdf5'
    TFILE = DDIR + '/mnist_test.hdf5'
    test_graph_one_shot_iterator_read(TFILE)
    test_eager_one_shot_iterator_read(TFILE)
