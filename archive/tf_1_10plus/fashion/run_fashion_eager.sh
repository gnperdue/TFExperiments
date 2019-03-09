#!/bin/bash
EXE="eager_fashion.py"

BATCH_SIZE=100
MODELDIR="/tmp/fashion_eager"
DATA_DIR="/Users/perdue/Dropbox/Data/RandomData/hdf5"

ARGS="--batch-size ${BATCH_SIZE} --model-dir $MODELDIR --data-dir ${DATA_DIR}"

python $EXE $ARGS
