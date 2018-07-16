#!/bin/bash

HDF5PATH="/Users/perdue/Dropbox/Data/RandomData/hdf5"
HDF5TRAIN="${HDF5PATH}/mnist_train.hdf5"
HDF5VALID="${HDF5PATH}/mnist_valid.hdf5"
HDF5TEST="${HDF5PATH}/mnist_test.hdf5"

TFRECPATH="/Users/perdue/Dropbox/Data/RandomData/TensorFlow"
TFRECTRAIN="${TFRECPATH}/mnist_train.tfrecord.gz"
TFRECVALID="${TFRECPATH}/mnist_valid.tfrecord.gz"
TFRECTEST="${TFRECPATH}/mnist_test.tfrecord.gz"

TRAINFILE="${HDF5TRAIN}"
VALIDFILE="${HDF5VALID}"

TRAINFILE="${TFRECTRAIN}"
VALIDFILE="${TFRECVALID}"

python logistic_regression.py \
  --train-file ${TRAINFILE} \
  --valid-file ${VALIDFILE} \
  -n 20
