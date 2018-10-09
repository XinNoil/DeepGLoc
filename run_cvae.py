#!/usr/bin/env python3
# coding: utf-8

import os,sys,math,argparse,ast
import pandas as pd
import numpy as np
import datasets as ds
from models import *

# Parsing arguments
parser = argparse.ArgumentParser(description='Training args')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--dataname', type=str, required=True)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--output', default='tmp')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--val',help='True or False flag, input should be either "True" or "False".',type=ast.literal_eval, default=True)
args = parser.parse_args()

# select gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

dataset = ds.load_data(args)
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
    epochs = args.epoch,
    batchsize = args.batchsize,
    reporter = ['loss'],
    validation = args.val
)

x_gen,y_sample = model.decodey(dataset.test_labels,resample_num=10)
gen_data = np.hstack((y_sample, x_gen))
np.savetxt('gen_data.csv', gen_data, fmt='%.3f',delimiter = ',')