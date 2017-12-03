#!/usr/bin/env python
"""
MNIST data in TFRecord form available at:
    https://github.com/gnperdue/RandomData/tree/master/TensorFlow

Example run script:
```
#!/bin/bash

FILEPAT="mnist"
DATADIR=/tmp/data
BATCHSIZE=10
ISIMAGE="True"

python min_examp.py --data_dir $DATADIR \
  --file_root $FILEPAT --compression "gz" \
  --is_image $ISIMAGE --batch_size $BATCHSIZE
```
Tested with Python2, TensorFlow 1.4
"""
from __future__ import print_function
from six.moves import range
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

ZLIB_COMP = tf.python_io.TFRecordCompressionType.ZLIB
GZIP_COMP = tf.python_io.TFRecordCompressionType.GZIP
NONE_COMP = tf.python_io.TFRecordCompressionType.NONE
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/tmp/data',
                           """Directory where data is stored.""")
tf.app.flags.DEFINE_string('file_root', 'mnist_',
                           """File basename.""")
tf.app.flags.DEFINE_string('compression', '',
                           """pigz (zz) or gzip (gz).""")
tf.app.flags.DEFINE_string('data_format', 'NHWC',
                           """Tensor packing structure.""")
tf.app.flags.DEFINE_string('out_pattern', 'temp_out',
                           """Logfile name.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size.""")
tf.app.flags.DEFINE_bool('is_image', False,
                         """Image formatting required.""")


def get_file_lists(data_dir, file_root, comp_ext):
    import glob
    comp_ext = comp_ext if comp_ext == '' else '.' + comp_ext
    train_list = glob.glob(data_dir + '/' + file_root +
                           '*_train.tfrecord' + comp_ext)
    valid_list = glob.glob(data_dir + '/' + file_root +
                           '*_valid.tfrecord' + comp_ext)
    test_list = glob.glob(data_dir + '/' + file_root +
                          '*_test.tfrecord' + comp_ext)
    if len(train_list) == 0 and \
       len(valid_list) == 0 and \
       len(test_list) == 0:
        raise IOError('No files found at specified path!')
    return train_list, valid_list, test_list


def make_data_reader_dict(
        filenames_list=None,
        batch_size=128,
        data_format='NHWC',
        compression=None,
        is_image=False
):
    data_reader_dict = {}
    data_reader_dict['FILENAMES_LIST'] = filenames_list
    data_reader_dict['BATCH_SIZE'] = batch_size
    data_reader_dict['DATA_FORMAT'] = data_format
    if compression is None:
        data_reader_dict['FILE_COMPRESSION'] = NONE_COMP
    elif compression == 'zz':
        data_reader_dict['FILE_COMPRESSION'] = ZLIB_COMP
    elif compression == 'gz':
        data_reader_dict['FILE_COMPRESSION'] = GZIP_COMP
    else:
        msg = 'Invalid compression type!'
        raise ValueError(msg)
    data_reader_dict['IS_IMG'] = is_image
    return data_reader_dict


def parse_mnist_tfrec(tfrecord, features_shape):
    tfrecord_features = tf.parse_single_example(
        tfrecord,
        features={
            'features': tf.FixedLenFeature([], tf.string),
            'targets': tf.FixedLenFeature([], tf.string)
        }
    )
    features = tf.decode_raw(tfrecord_features['features'], tf.uint8)
    features = tf.reshape(features, features_shape)
    features = tf.cast(features, tf.float32)
    targets = tf.decode_raw(tfrecord_features['targets'], tf.uint8)
    targets = tf.reshape(targets, [])
    targets = tf.one_hot(indices=targets, depth=10, on_value=1, off_value=0)
    targets = tf.cast(targets, tf.float32)
    return features, targets


class MNISTDataReaderDset:
    def __init__(self, data_reader_dict):
        self.filenames_list = data_reader_dict['FILENAMES_LIST']
        self.batch_size = data_reader_dict['BATCH_SIZE']
        self.data_format = data_reader_dict['DATA_FORMAT']
        compression = data_reader_dict['FILE_COMPRESSION']
        if compression == NONE_COMP:
            self.compression_type = ''
        elif compression == GZIP_COMP:
            self.compression_type = 'GZIP'
        elif compression == ZLIB_COMP:
            self.compression_type = 'ZLIB'
        else:
            raise ValueError('Unsupported compression: {}'.format(compression))
        self.is_image = data_reader_dict['IS_IMG']
        if self.is_image:
            self.features_shape = [28, 28, 1]
        else:
            self.features_shape = [784]

    def batch_generator(self, num_epochs=1):
        def parse_fn(tfrecord):
            return parse_mnist_tfrec(
                tfrecord, self.features_shape
            )
        dataset = tf.data.TFRecordDataset(
            self.filenames_list, compression_type=self.compression_type
        )
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.map(parse_fn).prefetch(self.batch_size)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
        return batch_features, batch_labels


def read_all_evtids(datareader_dict, typ):
    out_file = FLAGS.out_pattern + typ + '.txt'
    tf.reset_default_graph()
    n_evt = 0

    with tf.Graph().as_default() as g:
        with tf.Session(graph=g) as sess:
            data_reader = MNISTDataReaderDset(datareader_dict)
            batch_features, batch_labels = \
                data_reader.batch_generator(num_epochs=1)

            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                with open(out_file, 'ab+') as f:
                    # look at 2 batches only
                    for _ in range(2):
                        labels, feats = sess.run([
                            batch_labels, batch_features
                        ])
                        n_evt += len(labels)
                        mnist_data = zip(labels, feats)
                        for datum in mnist_data:
                            print(datum[0].shape)
                            print(datum[1].shape)
                            label = np.argmax(datum[0])
                            plt.imshow(datum[1].reshape((28, 28)))
                            fname = 'mnist_' + \
                                str(np.random.randint(1000000000)) + \
                                '_' + str(label) + '.png'
                            plt.savefig(fname, bbox_inches='tight')
                            f.write('{}\n'.format(datum[0]))
                            f.write('{}\n'.format(datum[1]))
            except tf.errors.OutOfRangeError:
                print('Reading stopped - queue is empty.')
            except Exception as e:
                print(e)
            finally:
                coord.request_stop()
                coord.join(threads)


def main(argv=None):
    train_list, valid_list, test_list = \
        get_file_lists(
            FLAGS.data_dir, FLAGS.file_root, FLAGS.compression
        )
    flist_dict = {}
    flist_dict['train'] = train_list
    flist_dict['valid'] = valid_list
    flist_dict['test'] = test_list

    for typ in ['train', 'valid', 'test']:
        dd = make_data_reader_dict(
            filenames_list=flist_dict[typ],
            batch_size=FLAGS.batch_size,
            compression=FLAGS.compression,
            is_image=FLAGS.is_image
        )
        read_all_evtids(dd, typ)


if __name__ == '__main__':
    tf.app.run()
