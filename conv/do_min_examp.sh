#!/bin/bash

FILEPAT="mnist"
DATADIR="${HOME}/Dropbox/Data/RandomData/TensorFlow/"
BATCHSIZE=10
ISIMAGE="True"

cat << EOF
python min_examp.py --data_dir $DATADIR \
  --file_root $FILEPAT --compression "gz" \
  --is_image $ISIMAGE --batch_size $BATCHSIZE
EOF

python min_examp.py --data_dir $DATADIR \
  --file_root $FILEPAT --compression "gz" \
  --is_image $ISIMAGE --batch_size $BATCHSIZE

