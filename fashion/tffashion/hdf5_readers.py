from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import h5py
import numpy as np


class FashionHDF5Reader:
    """
    user should call `openf()` and `closef()` to start/finish.

    assumes stored image shape is [N, depth(=1 greyscale), H, W]
    """

    def __init__(self, hdf5_file):
        self._file = hdf5_file
        self._f = None
        self._nlabels = 10
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]

    def openf(self):
        self._f = h5py.File(self._file, 'r')
        self._nevents = self._f['fashion/labels'].shape[0]
        return self._nevents

    def closef(self):
        try:
            self._f.close()
        except AttributeError:
            print('hdf5 file is not open yet.')

    def get_images(self, start_idx, stop_idx):
        tnsr = self._f['fashion/images'][start_idx: stop_idx]
        tnsr = np.moveaxis(tnsr, 1, -1)
        return tnsr

    def get_labels(self, start_idx, stop_idx):
        '''one-hot labels'''
        raw = self._f['fashion/labels'][start_idx: stop_idx].reshape([-1])
        ret = np.zeros((raw.size, self._nlabels), dtype=np.uint8)
        ret[np.arange(raw.size), raw] = 1
        return ret

    def get_example(self, idx):
        image = self._f['fashion/images'][idx]
        image = np.moveaxis(image, 0, -1)
        label = self._f['fashion/labels'][idx].reshape([-1])
        oh_label = np.zeros((1, self._nlabels), dtype=np.uint8)
        oh_label[0, label] = 1
        return image, oh_label.reshape(self._nlabels,)

    def get_flat_example(self, idx):
        image, label = self.get_example(idx)
        image = np.reshape(image, (28 * 28))
        return image, label
