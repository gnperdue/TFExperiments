from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import h5py
import numpy as np


class MNISTHDF5Reader:
    """
    user should call `openf()` and `closef()` to start/finish.

    assumes stored image shape is [N, depth, H, W]
    """
    def __init__(self, hdf5_file):
        self._file = hdf5_file
        self._f = None
        self._nlabels = 10

    def openf(self):
        self._f = h5py.File(self._file, 'r')
        self._nevents = self._f['targets'].shape[0]
        return self._nevents

    def closef(self):
        try:
            self._f.close()
        except AttributeError:
            print('hdf5 file is not open yet.')

    def get_features(self, start_idx, stop_idx):
        tnsr = self._f['features'][start_idx: stop_idx]
        tnsr = np.moveaxis(tnsr, 1, -1)
        return tnsr

    def get_labels(self, start_idx, stop_idx):
        raw = self._f['targets'][start_idx: stop_idx].reshape([-1])
        ret = np.zeros((raw.size, self._nlabels))
        ret[np.arange(raw.size), raw] = 1
        return ret

    def get_example(self, idx):
        feats = self._f['features'][idx]
        feats = np.moveaxis(feats, 0, -1)
        targs = self._f['targets'][idx].reshape([-1])
        oh_targs = np.zeros((1, self._nlabels))
        oh_targs[0, targs] = 1
        return feats, oh_targs.reshape(self._nlabels,)
