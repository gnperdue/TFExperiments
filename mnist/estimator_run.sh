#!/bin/bash

HDF5DIR="/Users/perdue/Dropbox/Data/RandomData/hdf5"

# if train-steps is empty, will train to num epochs
TRAINSTEPS=""
TRAINSTEPS="--train-steps 200"

python mnist_estimator.py --hdf5-dir $HDF5DIR $TRAINSTEPS
