from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf


def _make_mnist_generator_fn(hdf5_file_name, use_oned_data=False):
    """
    make a generator function that we can query for batches
    """
    from tfmnist.hdf5_readers import MNISTHDF5Reader as HDF5Reader
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


def make_mnist_hdf5dset(
        hdf5_file, batch_size, num_epochs, use_oned_data=False
):
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


def make_mnist_hdf5iterators(
        hdf5_file, batch_size, num_epochs, use_oned_data=False
):
    ds = make_mnist_hdf5dset(
        hdf5_file, batch_size, num_epochs, use_oned_data
    )

    # one_shot_iterators do not have initializers
    itrtr = ds.make_one_shot_iterator()
    feats, labs = itrtr.get_next()
    return feats, labs
    # return ds


def _parse_mnist_tfrec(tfrecord, features_shape):
    tfrecord_features = tf.parse_single_example(
        tfrecord,
        features={
            'features': tf.FixedLenFeature([], tf.string),
            'targets': tf.FixedLenFeature([], tf.string)
        },
        name='mnist_data'
    )
    features = tf.decode_raw(tfrecord_features['features'], tf.uint8)
    features = tf.reshape(features, features_shape)
    features = tf.cast(features, tf.float32)
    targets = tf.decode_raw(tfrecord_features['targets'], tf.uint8)
    targets = tf.one_hot(indices=targets, depth=10, on_value=1, off_value=0)
    targets = tf.reshape(targets, [10])
    targets = tf.cast(targets, tf.float32)
    return features, targets


def make_mnist_tfrecdset(
        tfrecfile, batch_size, num_epochs, use_oned_data=False, shuffle=False
):
    if use_oned_data:
        features_shape = [784]
    else:
        features_shape = [28, 28, 1]

    def parse_fn(tfrecord):
        return _parse_mnist_tfrec(tfrecord, features_shape)

    dataset = tf.data.TFRecordDataset([tfrecfile], compression_type='GZIP')
    if shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(parse_fn).prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    return dataset


def make_mnist_tfreciterators(
        tfrecfile, batch_size, num_epochs, use_oned_data=False, shuffle=False
):
    dataset = make_mnist_tfrecdset(
        tfrecfile, batch_size, num_epochs, use_oned_data, shuffle
    )
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def _get_test_hdf5():
    import os
    DDIR = os.environ['HOME'] + '/Dropbox/Data/RandomData/hdf5'
    TFILE = DDIR + '/mnist_test.hdf5'
    return TFILE


def _get_test_tfrec():
    import os
    DDIR = os.environ['HOME'] + '/Dropbox/Data/RandomData/TensorFlow'
    TFILE = DDIR + '/mnist_valid.tfrecord.gz'
    return TFILE


def test_graph_one_shot_iterator(
        mkiter_fn=make_mnist_hdf5iterators, getdata_fn=_get_test_hdf5
):
    data_file = getdata_fn()
    batch_size = 25
    num_epochs = 1
    feats, labs = mkiter_fn(
        data_file, batch_size, num_epochs
    )
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


def test_graph_one_shot_iterator_hdf5read():
    getdata_fn = _get_test_hdf5
    mkiter_fn = make_mnist_hdf5iterators
    test_graph_one_shot_iterator(mkiter_fn=mkiter_fn, getdata_fn=getdata_fn)


def test_graph_one_shot_iterator_tfrecread():
    getdata_fn = _get_test_tfrec
    mkiter_fn = make_mnist_tfreciterators
    test_graph_one_shot_iterator(mkiter_fn=mkiter_fn, getdata_fn=getdata_fn)


def test_eager_one_shot_iterator(mkdset_fn, getdata_fn):
    data_file = getdata_fn()
    batch_size = 25
    num_epochs = 1
    tfe = tf.contrib.eager
    tf.enable_eager_execution()
    targets_and_labels = mkdset_fn(
        data_file, batch_size, num_epochs, use_oned_data=True
    )

    for i, (fs, ls) in enumerate(tfe.Iterator(targets_and_labels)):
        print(fs.shape, fs.dtype, ls.shape, ls.dtype)


def test_eager_one_shot_iterator_hdf5read():
    getdata_fn = _get_test_hdf5
    mkdset_fn = make_mnist_hdf5dset
    test_eager_one_shot_iterator(mkdset_fn=mkdset_fn, getdata_fn=getdata_fn)


def test_eager_one_shot_iterator_tfrecread():
    getdata_fn = _get_test_tfrec
    mkdset_fn = make_mnist_tfrecdset
    test_eager_one_shot_iterator(mkdset_fn=mkdset_fn, getdata_fn=getdata_fn)
