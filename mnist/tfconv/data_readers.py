from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf


def _make_mnist_generator_fn(hdf5_file_name):
    """
    make a generator function that we can query for batches
    """
    from tfconv.hdf5_readers import MNISTHDF5Reader as HDF5Reader
    reader = HDF5Reader(hdf5_file_name)
    nevents = reader.openf()

    def example_generator_fn():
        idx = 0
        while True:
            if idx >= nevents:
                return
            yield reader.get_example(idx)
            idx += 1
            
        reader.closef()

    return example_generator_fn


def make_mnist_iterators(hdf5_file, batch_size, num_epochs):
    # make a generator function
    dgen = _make_mnist_generator_fn(hdf5_file)

    # make a Dataset from a generator
    features_shape = [28, 28, 1]
    labels_shape = [10]
    ds = tf.data.Dataset.from_generator(
        dgen, (tf.float32, tf.uint8),
        (tf.TensorShape(features_shape), tf.TensorShape(labels_shape))
    )
    ds = ds.shuffle(1000).batch(batch_size).repeat(num_epochs)

    # one_shot_iterators do not have initializers
    itrtr = ds.make_one_shot_iterator()
    feats, labs = itrtr.get_next()
    return feats, labs
    # return ds


def test_one_shot_iterator_read(hdf5_file, batch_size=25, num_epochs=1):
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


if __name__ == '__main__':
    import os
    # Get path to data
    DDIR = os.environ['HOME'] + '/Dropbox/Data/RandomData/hdf5'
    TFILE = DDIR + '/mnist_test.hdf5'
    test_one_shot_iterator_read(TFILE)
