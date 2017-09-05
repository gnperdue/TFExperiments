#!/bin/bash

NTRAINBATCH=20
if [ $# -gt 0 ]; then
  NTRAINBATCH=$1
fi

DATADIR="${HOME}//Dropbox/Data/RandomData/TensorFlow/"
MODELDIR="/tmp/models"

python logistic_regression.py -n $NTRAINBATCH -m $MODELDIR -d $DATADIR
