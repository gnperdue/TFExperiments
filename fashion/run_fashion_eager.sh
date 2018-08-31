#!/bin/bash
EXE="eager_fashion.py"

BATCH_SIZE=100
NUM_EPOCHS=2
MODELDIR="/tmp/fashion_eager"
DATA_DIR="/Users/perdue/Dropbox/Data/RandomData/hdf5"

ARGS="--batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --model-dir $MODELDIR --data-dir ${DATA_DIR}"

python $EXE $ARGS
