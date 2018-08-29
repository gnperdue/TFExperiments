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


ARGS="--batch_size ${BATCH_SIZE} --train_steps ${TRAIN_STEPS} --num_epochs ${NUM_EPOCHS} --data_dir ${DATA_DIR} $TFRECORD --model_dir $MODELDIR"

python $EXE $ARGS
