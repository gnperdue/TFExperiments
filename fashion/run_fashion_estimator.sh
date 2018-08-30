#!/bin/bash
EXE="estimator_fashion.py"
BATCH_SIZE=100
TRAIN_STEPS=50
NUM_EPOCHS=2
MODELDIR="/tmp/fashion_estimator"


DATA_DIR="/Users/perdue/Dropbox/Data/RandomData/TensorFlow"
TFRECORD="--tfrecord"

DATA_DIR="/Users/perdue/Dropbox/Data/RandomData/hdf5"
TFRECORD=""

ARGS="--batch-size ${BATCH_SIZE} --train-steps ${TRAIN_STEPS} --num-epochs
${NUM_EPOCHS} --data-dir ${DATA_DIR} $TFRECORD --model-dir $MODELDIR"

python $EXE $ARGS
