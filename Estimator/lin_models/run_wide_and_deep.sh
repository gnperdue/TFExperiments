#!/bin/bash

MODELTYPE="wide"
TRAINSTEPS=100

python wide_and_deep.py \
       --model_dir /tmp/wide_and_deep \
       --model_type $MODELTYPE \
       --train_steps $TRAINSTEPS
