#!/bin/bash

HDF5DIR="/Users/perdue/Dropbox/Data/RandomData/hdf5"

# if train-steps is empty, will train to num epochs
TRAINSTEPS=""
TRAINSTEPS="--train-steps 10"

python mnist_graph.py --hdf5-dir $HDF5DIR $TRAINSTEPS
