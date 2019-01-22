TensorFlow experiments.

The MNIST and Fashion MNIST HDF5 files used here are available in this
[GitHub Project](https://github.com/gnperdue/RandomData). You can download
directly from the repository without checking out the whole repository
using `wget` or `curl`, e.g.

* `wget https://github.com/gnperdue/RandomData/tree/master/hdf5/mnist_test.hdf5`
* `wget https://github.com/gnperdue/RandomData/tree/master/hdf5/mnist_train.hdf5`
* `wget https://github.com/gnperdue/RandomData/tree/master/hdf5/mnist_valid.hdf5`
* `wget https://github.com/gnperdue/RandomData/tree/master/hdf5/fashion_test.hdf5`
* `wget https://github.com/gnperdue/RandomData/tree/master/hdf5/fashion_train.hdf5`

etc. (browse the directory for more, e.g., a breakdown of the MNIST set by digits).
There you will also find some useful scripts for visualization, producing the HDF5
files, etc. You may also download `TFRecords` for use with TensorFlow, e.g.

* `wget https://github.com/gnperdue/RandomData/tree/master/TensorFlow/mnist_test.tfrecord.gz`

To parse the `TFRecords`, I recommend reading the 
[writer script](https://github.com/gnperdue/RandomData/blob/master/TensorFlow/mnist_hdf5_to_tfrec.py),
which explains how the `TFRecords` are structured. 
