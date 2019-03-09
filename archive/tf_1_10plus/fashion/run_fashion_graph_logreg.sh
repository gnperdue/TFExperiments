#!/bin/bash
EXE="graph_logreg_fashion.py"

TRAIN_STEPS=20
BATCH_SIZE=100
MODELDIR="/tmp/graph_logreg"
DATA_DIR="/Users/perdue/Dropbox/Data/RandomData/hdf5"

ARGS="--model-dir $MODELDIR --data-dir ${DATA_DIR} --train-steps ${TRAIN_STEPS}"

cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS
