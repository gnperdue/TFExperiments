#!/bin/bash

DAT=`date +%s`

BASEP="${HOME}/Documents/MINERvA/AI/mnist_tf"
FILEPAT="mnist"
DATADIR="${HOME}/Dropbox/Data/RandomData/TensorFlow/"
LOGDIR="${BASEP}/logs"
LOGFILE=$LOGDIR/log_tfrecex_mnist${DAT}.txt
BATCHSIZE=10

ISIMAGE="False"

cat << EOF
python tfrec_examiner.py --data_dir $DATADIR \
  --file_root $FILEPAT --compression "gz" \
  --log_name $LOGFILE --is_image $ISIMAGE \
  --batch_size $BATCHSIZE
EOF

python tfrec_examiner.py --data_dir $DATADIR \
  --file_root $FILEPAT --compression "gz" \
  --log_name $LOGFILE --is_image $ISIMAGE \
  --batch_size $BATCHSIZE
