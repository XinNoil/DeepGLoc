#!/usr/bin/env python3
# coding: utf-8

import os,sys,math,argparse
import pandas as pd
import numpy as np
import datasets as ds
from models import *

models = {
    'dnn': BaseDNN,
    'vae': VAE,
    'cvae': CVAE
}

# Parsing arguments
parser = argparse.ArgumentParser(description='Training args')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--output', default='output')
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()

# select gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

dataset=ds.load_data(args)
args.val=dataset.val
dataset.get_validation()
# Construct model
if args.model not in models:
    raise Exception('Unknown model:', args.model)

model = models[args.model](
    input_shape=dataset.get_input_shape(),
    output_shape=dataset.get_output_shape(),
    output=args.output
)

model.main_loop(dataset,
    epochs=args.epoch,
    batchsize=args.batchsize,
    reporter=['loss'],
    validation=args.val
)
