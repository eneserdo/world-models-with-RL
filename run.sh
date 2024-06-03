#!/bin/bash

LOGDIR=$1

echo "Using $LOGDIR as logdir"
if [ -z "$LOGDIR" ]; then
    echo "Usage: run.sh LOGDIR"
    exit 1
fi

mkdir -p ${LOGDIR}

echo "### Generating dataset ###"
python data/generation_script.py --rollouts 1000 --rootdir datasets/carracing --threads 8
sleep 5

echo "### Training VAE ###"
python trainvae.py --logdir ${LOGDIR} --dataset_dir datasets/carracing
sleep 5

echo "### Training MDRNN ###"
python trainmdrnn.py --logdir ${LOGDIR} --dataset_dir datasets/carracing
sleep 5

echo "### Training Controller ###"
# python traincontroller.py --logdir ${LOGDIR} --n-samples 4 --pop-size 4 --target-return 950 --display --dataset_dir datasets/carracing
xvfb-run -s "-screen 0 1400x900x24" python traincontroller.py --logdir ${LOGDIR} --n-samples 4 --pop-size 4 --target-return 950 --display --dataset_dir datasets/carracing