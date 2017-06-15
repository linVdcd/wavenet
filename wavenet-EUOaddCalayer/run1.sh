#!/bin/sh


CUDA_VISIBLE_DEVICES=2 python train.py --wavenet_params=wavenet_params_1.json --learning_rate=0.001 --num_steps=100000 --logdir=./logdir/train/2016-12-30T20-28-13
CUDA_VISIBLE_DEVICES=2 python train.py --wavenet_params=wavenet_params_1.json --learning_rate=0.0001 --logdir=./logdir/train/2016-12-30T20-28-13

