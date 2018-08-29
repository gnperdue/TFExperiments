#!/usr/bin/env python
import h5py
import logging
import os

LOGGER = logging.getLogger(__name__)
DATA_PATH = os.environ['HOME'] + '/Dropbox/Data/RandomData/hdf5/'


class MNISTHDF5Reader:
    """
    mnist hdf5 files has `features` and `targets` w/ shapes (70000, 1, 28, 28)
    and (70000, 1) and dtype (both) uint8. the `mnist_hdf5_reader` will return
    numpy ndarrays of data for given ranges. user should call `open()` and
    `close()` to start/finish.
    """
    def __init__(self, hdf5_file):
        self.file = hdf5_file
        self._f = None

    def open(self):
        LOGGER.info('opening hdf5 file {}'.format(self.file))
        self._f = h5py.File(self.file, 'r')

    def close(self):
        try:
            LOGGER.info('closing hdf5 file {}'.format(self.file))
            self._f.close()
        except AttributeError:
            print('hdf5 file is not open yet.')
            LOGGER.error('hdf5 file is not open yet.')

    def get_features(self, start_idx, stop_idx):
        return self._f['features'][start_idx: stop_idx]

    def get_targets(self, start_idx, stop_idx):
        return self._f['targets'][start_idx: stop_idx]
