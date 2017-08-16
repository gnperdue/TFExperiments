"""
mnist data
"""
import logging
import tensorflow as tf
import numpy as np
import os

LOGGER = logging.getLogger(__name__)
DATA_PATH = os.environ['HOME'] + '/Dropbox/Data/RandomData/TensorFlow/'
BATCH_SIZE = 20


def tfrecord_to_graph_ops(filenames_list, num_epochs=1):
    with tf.variable_scope('tfrec_to_graph'):
        file_queue = tf.train.string_input_producer(
            filenames_list, name='file_queue', num_epochs=num_epochs
        )
        reader = tf.TFRecordReader(
            options=tf.python_io.TFRecordOptions(
                compression_type=tf.python_io.TFRecordCompressionType.GZIP
            ), name='tfrec_reader'
        )
        _, tfrecord = reader.read(file_queue)

        tfrecord_features = tf.parse_single_example(
            tfrecord,
            features={
                'features': tf.FixedLenFeature([], tf.string),
                'targets': tf.FixedLenFeature([], tf.string)
            },
            name='data'
        )
        with tf.variable_scope('features'):
            features = tf.decode_raw(tfrecord_features['features'], tf.uint8)
            features = tf.reshape(features, [-1, 784])
            features = tf.cast(features, tf.float32)
        with tf.variable_scope('targets'):
            targets = tf.decode_raw(tfrecord_features['targets'], tf.uint8)
            targets = tf.cast(targets, tf.int32)
            targets = tf.reshape(targets, [-1])
            targets = tf.one_hot(
                indices=targets, depth=10, on_value=1, off_value=0
            )
        return features, targets


def batch_generator(filenames_list, batch_size=BATCH_SIZE, num_epochs=1):
    with tf.variable_scope('batchgen'):
        features, targets = tfrecord_to_graph_ops(filenames_list, num_epochs)
        min_after_dequeue = 3 * batch_size
        capacity = 20 * batch_size
        features_batch, targets_batch = tf.train.shuffle_batch(
            [features, targets],
            batch_size=batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            enqueue_many=True
        )
    return features_batch, targets_batch


def examine_batches(features_batch, targets_batch):
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            for it in range(5000):
                features, targets = sess.run([features_batch, targets_batch])
                if it % 100 == 0:
                    LOGGER.debug(it)
                    LOGGER.debug(
                        len(features),
                        features[0].shape,
                        np.max(features[0][0][7][:])
                    )
                    LOGGER.debug(np.argmax(targets, axis=1))
        except tf.errors.OutOfRangeError:
            LOGGER.info('Training stopped - queue is empty.')
        except Exception as e:
            LOGGER.error(e)
        finally:
            coord.request_stop()
            coord.join(threads)


def test():
    test_file = DATA_PATH + 'mnist_test.tfrecord.gz'
    if os.path.isfile(test_file):
        file_list = [test_file]
    else:
        raise Exception('No data file!')
    features_batch, targets_batch = batch_generator(
        file_list, batch_size=5, num_epochs=1
    )
    examine_batches(features_batch, targets_batch)


if __name__ == '__main__':
    test()
