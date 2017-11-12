#!/usr/bin/env python
import tensorflow as tf
import utils_mnist


def parse_mnist_tfrec(tfrecord, name, features_shape):
    tfrecord_features = tf.parse_single_example(
        tfrecord,
        features={
            'features': tf.FixedLenFeature([], tf.string),
            'targets': tf.FixedLenFeature([], tf.string)
        },
        name=name+'_data'
    )
    with tf.variable_scope('features'):
        features = tf.decode_raw(
            tfrecord_features['features'], tf.uint8
        )
        features = tf.reshape(features, features_shape)
        features = tf.cast(features, tf.float32)
    with tf.variable_scope('targets'):
        targets = tf.decode_raw(tfrecord_features['targets'], tf.uint8)
        targets = tf.reshape(targets, [-1])
        targets = tf.one_hot(
            indices=targets, depth=10, on_value=1, off_value=0
        )
        targets = tf.cast(targets, tf.float32)
    return features, targets


class MNISTDataReader:
    """
    name values are usually, e.g., 'train' or 'validation', etc.

    allowed compression values:
    * `tf.python_io.TFRecordCompressionType.ZLIB`
    * `tf.python_io.TFRecordCompressionType.GZIP`
    """
    def __init__(self, data_reader_dict):
        self.filenames_list = data_reader_dict['FILENAMES_LIST']
        self.batch_size = data_reader_dict['BATCH_SIZE']
        self.name = data_reader_dict['NAME']
        self.data_format = data_reader_dict['DATA_FORMAT']
        self.compression = data_reader_dict['FILE_COMPRESSION']
        self.is_image = data_reader_dict['IS_IMG']
        if self.is_image:
            self.features_shape = [-1, 28, 28, 1]
        else:
            self.features_shape = [-1, 784]

    def _tfrecord_to_graph_ops(self, num_epochs):
        with tf.variable_scope('tfrec_to_graph'):
            file_queue = tf.train.string_input_producer(
                self.filenames_list,
                name=self.name+'_file_queue',
                num_epochs=num_epochs
            )
            reader = tf.TFRecordReader(
                options=tf.python_io.TFRecordOptions(
                    compression_type=self.compression
                ), name=self.name+'_tfrec_reader'
            )
            _, tfrecord = reader.read(file_queue)

            features, targets = parse_mnist_tfrec(
                tfrecord, self.name, self.features_shape
            )
        return features, targets

    def batch_generator(self, num_epochs=1):
        with tf.variable_scope(self.name+'_batch_gen'):
            features, targets = self._tfrecord_to_graph_ops(num_epochs)
            capacity = 10 * self.batch_size
            features_batch, targets_batch = tf.train.batch(
                [features, targets],
                batch_size=self.batch_size,
                capacity=capacity,
                enqueue_many=True,
                allow_smaller_final_batch=True,
                name=self.name+'_batch'
            )
        return features_batch, targets_batch

    def shuffle_batch_generator(self, num_epochs=1):
        with tf.variable_scope(self.name+'_shufflebatch_gen'):
            features, targets = self._tfrecord_to_graph_ops(num_epochs)
            min_after_dequeue = 3 * self.batch_size
            capacity = 10 * self.batch_size
            features_batch, targets_batch = tf.train.shuffle_batch(
                [features, targets],
                batch_size=self.batch_size,
                capacity=capacity,
                min_after_dequeue=min_after_dequeue,
                enqueue_many=True,
                allow_smaller_final_batch=True,
                name=self.name+'_shuffle_batch'
            )
        return features_batch, targets_batch


class MNISTDataReaderDset:
    """ etc. """
    def __init__(self, data_reader_dict):
        self.filenames_list = data_reader_dict['FILENAMES_LIST']
        self.batch_size = data_reader_dict['BATCH_SIZE']
        self.name = data_reader_dict['NAME']
        self.data_format = data_reader_dict['DATA_FORMAT']
        compression = data_reader_dict['FILE_COMPRESSION']
        if compression == utils_mnist.NONE_COMP:
            self.compression_type = ''
        elif compression == utils_mnist.GZIP_COMP:
            self.compression_type = 'GZIP'
        elif compression == utils_mnist.ZLIB_COMP:
            self.compression_type = 'ZLIB'
        else:
            raise ValueError('Unsupported compression: {}'.format(compression))
        self.is_image = data_reader_dict['IS_IMG']
        if self.is_image:
            self.features_shape = [-1, 28, 28, 1]
            # self.features_shape = [28, 28, 1]
        else:
            self.features_shape = [-1, 784]
            # self.features_shape = [784]

    def shuffle_batch_generator(self, num_epochs=1):
        pass

    def batch_generator(self, num_epochs=1):
        """
        TODO - we can use placeholders for the list of file names and
        init with a feed_dict when we call `sess.run` - give this a
        try with one list for training and one for validation
        """
        def parse_fn(tfrecord):
            return parse_mnist_tfrec(
                tfrecord, self.name, self.features_shape
            )
        dataset = tf.data.TFRecordDataset(
            self.filenames_list, compression_type=self.compression_type
        )
        dataset = dataset.map(parse_fn)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
        return batch_features, batch_labels
