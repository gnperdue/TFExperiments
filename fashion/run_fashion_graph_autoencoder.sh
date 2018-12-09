#!/bin/bash
EXE="graph_autoencoder_fashion.py"

TRAIN_STEPS=250
BATCH_SIZE=100
MODELDIR="/tmp/graph_autoencoder"
DATA_DIR="/Users/perdue/Dropbox/Data/RandomData/hdf5"

ARGS="--model-dir $MODELDIR"
ARGS+=" --data-dir ${DATA_DIR}"
#ARGS+=" --train-steps ${TRAIN_STEPS}"

cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS
