#!/bin/bash

SNGLRTY="/data/perdue/singularity/simone/ubuntu16-cuda-ml.img"

NTRAINBATCH=20
if [ $# -gt 0 ]; then
  NTRAINBATCH=$1
fi

DATADIR="/data/perdue/mnist_tf/data/"
MODELDIR="/data/perdue/mnist_tf/models"

cat << EOF
singularity exec $SNGLRTY python logistic_regression.py \
  -n $NTRAINBATCH \
  -m $MODELDIR \
  -d $DATADIR
EOF

singularity exec $SNGLRTY python logistic_regression.py \
  -n $NTRAINBATCH \
  -m $MODELDIR \
  -d $DATADIR
