#!/usr/bin/env python3
# coding: utf-8

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os,sys,math,argparse,ast,utils
import pandas as pd
import numpy as np
import datasets as ds
import tensorflow as tf
from models import *

# Parsing arguments
parser = argparse.ArgumentParser(description='Training args')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--dataname', type=str, required=True)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--output', default='tmp')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--val',help='True or False flag, input should be either "True" or "False".',type=ast.literal_eval, default=True)
parser.add_argument('--gen',help='True or False flag, input should be either "True" or "False".',type=ast.literal_eval, default=True)
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

predict_result=model.predict(dataset.test_features)
loss=utils.euclidean_error(dataset.test_labels, predict_result)
sess=tf.Session()
loss_np=sess.run(loss)
loss_np=np.reshape(loss_np,(dataset.test_labels.shape[0],1))
print('Test loss:\t',np.mean(loss_np),'\n')

utils.cdfplot(loss_np)
plt.savefig(model.get_filename(utils.get_prefix(args),'losscdf','png'))