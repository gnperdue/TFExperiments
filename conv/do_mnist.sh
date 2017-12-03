#!/bin/bash

DAT=`date +%s`

TRAINING="--nodo_training"
TRAINING="--do_training"
VALIDATION="--do_validaton"

TESTING="--nodo_testing"
TESTING="--do_testing"

CONV="--nodo_conv"
CONV="--do_conv"

NEPOCHS="--num_epochs 3"
NEPOCHS="--num_epochs 1"

SHORT=""
SHORT="--do_a_short_run"

BATCHNORM="--do_batch_norm"
BATCHNORM=""

LRATE="--learning_rate 0.05"
LRATE="--learning_rate 0.01"

MODEL_CODE="20171017_logreg"
MODEL_CODE="20171017_batch"
MODEL_CODE="20171017_nobatch"

# data, log, and model logistics
BASEP="${HOME}/Documents/MINERvA/AI/mnist_tf"
FILEPAT="mnist"
DATADIR="${HOME}/Dropbox/Data/RandomData/TensorFlow/"
LOGDIR="${BASEP}/logs"
LOGFILE=$LOGDIR/log_mnv_st_epsilon_${NCLASS}_${MODEL_CODE}_${DAT}.txt
LOGLEVEL="--log_level DEBUG"
LOGLEVEL="--log_level INFO"
MODELDIR="${BASEP}/models/${NCLASS}/${MODEL_CODE}"

# show what we will do...
cat << EOF
python run_mnist.py \
  $CONV \
  --compression gz \
  --data_dir $DATADIR \
  --file_root $FILEPAT \
  --model_dir $MODELDIR \
  --log_name $LOGFILE $LOGLEVEL \
  $TRAINING $VALIDATION $TESTING $SHORT \
  $BATCHNORM $NEPOCHS $LRATE
EOF

python run_mnist.py \
  $CONV \
  --compression gz \
  --data_dir $DATADIR \
  --file_root $FILEPAT \
  --model_dir $MODELDIR \
  --log_name $LOGFILE $LOGLEVEL \
  $TRAINING $VALIDATION $TESTING $SHORT \
  $BATCHNORM $NEPOCHS $LRATE

echo "Job finished "`date`""
