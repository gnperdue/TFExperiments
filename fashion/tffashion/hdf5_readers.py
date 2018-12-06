from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import h5py
import numpy as np


class FashionHDF5Reader(object):
    """
    user should call `openf()` and `closef()` to start/finish.

    assumes stored image shape is [N, depth(=1 greyscale), H, W]
    """

    def __init__(self, hdf5_file, tofloat=False):
        self._file = hdf5_file
        self._f = None
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        self._nlabels = len(self.class_names)
        self._tofloat = tofloat

    def openf(self):
        self._f = h5py.File(self._file, 'r')
        self._nevents = self._f['fashion/labels'].shape[0]
        return self._nevents

    def closef(self):
        try:
            self._f.close()
        except AttributeError:
            print('hdf5 file is not open yet.')

    def get_example(self, idx):
        image = self._f['fashion/images'][idx]
        image = np.moveaxis(image, 0, -1)
        label = self._f['fashion/labels'][idx].reshape([-1])
        oh_label = np.zeros((1, self._nlabels), dtype=np.uint8)
        oh_label[0, label] = 1
        if self._tofloat:
            return image.astype(np.float32), \
                oh_label.reshape(self._nlabels,).astype(np.float32)
        return image, oh_label.reshape(self._nlabels,)

    def get_flat_example(self, idx):
        image, label = self.get_example(idx)
        image = np.reshape(image, (28 * 28))
        return image, label

    def get_examples(self, start_idx, stop_idx):
        image = self._f['fashion/images'][start_idx: stop_idx]
        image = np.moveaxis(image, 1, -1)
        label = self._f['fashion/labels'][start_idx: stop_idx].reshape([-1])
        oh_label = np.zeros((label.size, self._nlabels), dtype=np.uint8)
        oh_label[np.arange(label.size), label] = 1
        if self._tofloat:
            return image.astype(np.float32), oh_label.astype(np.float32)
        return image, oh_label

    def get_flat_examples(self, start_idx, stop_idx):
        image, label = self.get_examples(start_idx, stop_idx)
        image = np.reshape(image, (-1, 28 * 28))
        return image, label
