#!/bin/sh


CUDA_VISIBLE_DEVICES=0 python train.py --learning_rate=0.001 --num_steps=100000 --logdir=logdir/train/2017-01-06T16-09-14
CUDA_VISIBLE_DEVICES=0 python train.py --learning_rate=0.0001 --num_steps=300000 --logdir=logdir/train/2017-01-06T16-09-14

