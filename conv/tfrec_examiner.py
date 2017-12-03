"""
"""
from __future__ import print_function
from six.moves import range
import tensorflow as tf
import logging
import os
import gzip
import shutil

from DataReaders import MNISTDataReaderTFRecDset as DataReader
# from DataReaders import MNISTDataReader as DataReader
import utils_mnist

LOGGER = logging.getLogger(__name__)
FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('data_dir', '/tmp/data',
                           """Directory where data is stored.""")
tf.app.flags.DEFINE_string('file_root', 'mnist_',
                           """File basename.""")
tf.app.flags.DEFINE_string('compression', '',
                           """pigz (zz) or gzip (gz).""")
tf.app.flags.DEFINE_string('data_format', 'NHWC',
                           """Tensor packing structure.""")
tf.app.flags.DEFINE_string('log_name', 'temp_log.txt',
                           """Logfile name.""")
tf.app.flags.DEFINE_string('out_pattern', 'temp_out',
                           """Logfile name.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size.""")
tf.app.flags.DEFINE_bool('is_image', False,
                         """Image formatting required.""")


def compress(out_file):
    gzfile = out_file + '.gz'
    with open(out_file, 'rb') as f_in, gzip.open(gzfile, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    if os.path.isfile(gzfile) and (os.stat(gzfile).st_size > 0):
        os.remove(out_file)
    else:
        raise IOError('Compressed file not produced!')


def read_all_evtids(datareader_dict, typ):
    LOGGER.info('read all labels for {}...'.format(typ))
    out_file = FLAGS.out_pattern + typ + '.txt'
    tf.reset_default_graph()
    n_evt = 0

    with tf.Graph().as_default() as g:
        with tf.Session(graph=g) as sess:
            data_reader = DataReader(datareader_dict)
            batch_features, batch_labels = \
                data_reader.batch_generator(num_epochs=1)

            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                with open(out_file, 'ab+') as f:
                    for _ in range(3):
                        labels, feats = sess.run([
                            batch_labels, batch_features
                        ])
                        n_evt += len(labels)
                        mnist_data = zip(labels, feats)
                        for datum in mnist_data:
                            LOGGER.info('labels.shape = {}'.format(
                                datum[0].shape
                            ))
                            LOGGER.info('features.shape = {}'.format(
                                datum[1].shape
                            ))
                            f.write('{}\n'.format(datum[0]))
                            f.write('{}\n'.format(datum[1]))
            except tf.errors.OutOfRangeError:
                LOGGER.info('Reading stopped - queue is empty.')
            except Exception as e:
                LOGGER.info(e)
            finally:
                coord.request_stop()
                coord.join(threads)

    LOGGER.info('found {} {} events'.format(n_evt, typ))
    compress(out_file)


def main(argv=None):
    logfilename = FLAGS.log_name
    logging.basicConfig(
        filename=logfilename, level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    LOGGER.info("Starting...")
    LOGGER.info(__file__)

    flist_dict = utils_mnist.get_file_lists(
        FLAGS.data_dir, FLAGS.file_root, FLAGS.compression
    )
    for typ in flist_dict.keys():
        dd = utils_mnist.make_data_reader_dict(
            filenames_list=flist_dict[typ],
            batch_size=FLAGS.batch_size,
            compression=FLAGS.compression,
            is_image=FLAGS.is_image
        )
        LOGGER.info(' data reader dict for {} = {}'.format(
            typ, repr(dd)
        ))
        read_all_evtids(dd, typ)


if __name__ == '__main__':
    tf.app.run()
