from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import logging
import argparse

import tensorflow as tf

from tffashion.data_readers import make_fashion_dset
from tffashion.data_readers import make_fashion_iterators

# Get path to data
TFILE = os.path.join(
    os.environ['HOME'], 'Dropbox/Data/RandomData/hdf5/fashion_test.hdf5'
)

logfilename = 'log_' + __file__.split('.')[0] + '.txt'
logging.basicConfig(
    filename=logfilename, level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("Starting...")
logger.info(__file__)
logger.info(' Examining file {}'.format(TFILE))


def test_graph_one_shot_iterator_read(
    hdf5_file=TFILE, batch_size=25, num_epochs=1
):
    feats, labs = make_fashion_iterators(hdf5_file, batch_size, num_epochs)
    with tf.Session() as sess:
        total_batches = 0
        total_examples = 0
        try:
            while True:
                fs, ls = sess.run([feats, labs])
                logger.info('{}, {}, {}, {}'.format(
                    fs.shape, fs.dtype, ls.shape, ls.dtype
                ))
                total_batches += 1
                total_examples += ls.shape[0]
                if total_batches > 1000:
                    break
        except tf.errors.OutOfRangeError:
            logger.info('end of dataset at total_batches={}'.format(
                total_batches
            ))
        except Exception as e:
            logger.error(e)
    logger.info('saw {} total examples'.format(total_examples))


def test_eager_one_shot_iterator_read(
    hdf5_file=TFILE, batch_size=25, num_epochs=1
):
    tfe = tf.contrib.eager
    tf.enable_eager_execution()
    targets_and_labels = make_fashion_dset(
        hdf5_file, batch_size, num_epochs, use_oned_data=True
    )

    total_examples = 0
    for i, (fs, ls) in enumerate(tfe.Iterator(targets_and_labels)):
        logger.info('{}, {}, {}, {}'.format(
            fs.shape, fs.dtype, ls.shape, ls.dtype
        ))
        total_examples += ls.shape[0]
    logger.info('saw {} total examples'.format(total_examples))


def main(eager, batch_size):
    if eager:
        test_eager_one_shot_iterator_read(batch_size=batch_size)
    else:
        test_graph_one_shot_iterator_read(batch_size=batch_size)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--eager', default=False, action='store_true',
        help='Use Eager execution'
    )
    parser.add_argument(
        '--batch-size', type=int, default=10,
        help='Batch size'
    )
    args = parser.parse_args()
    main(**vars(args))
