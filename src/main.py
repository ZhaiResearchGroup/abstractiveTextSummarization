import argparse
import os
import random
import torch
import torch.nn as nn
import numpy as np
from torch import cuda
from utils.dataloader import Dataloader
from model.seq2seq import Seq2Seq
from train.trainer import Trainer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


## Data options
parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-teacher_forcing_ratio', type=float, default=0.6,help='Probablity of using teacher forcing (scheduled sampling)')
parser.add_argument('-trnd', '--traindata', default='../data/wiki_queries12_head.csv', help='Path to train data file')
parser.add_argument('-tstd', '--testdata', default='../data/wiki_queries12_head.csv', help="Path to the test data file")
parser.add_argument('-pd', '--processeddata', default='dataset/data.pkl', help="Path to the pre-processed data set")
parser.add_argument('-sos', '--sos_token', default="<sos>", help='Adding EOS token at the end of each sequence')
parser.add_argument('-sdir', '--save_dir', default='saving', help='Directory to save model checkpoints')
parser.add_argument('-ldir', '--load_dir', default='loading', help='Path to a model checkpoint')
parser.add_argument('-vs', "--vocab_size", type=int, default=None, help="Limit vocabulary")
parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch Size for seq2seq model')
parser.add_argument('-gpu', type=int, default=-1, help='GPU id. Support single GPU only')
parser.add_argument('-ne', '--n_epochs', type=int, default=1,help='number of training epochs')
parser.add_argument('-lr', '--learning_rate', type=float, default=.0001, help='learning rate')

opt = parser.parse_args()

opt.cuda = (opt.gpu != -1)

dataloader = Dataloader(opt)
print ('Done loading data')

model = Seq2Seq(opt, dataloader.vocab)
print ('Done building model')

trainer = Trainer(opt, dataloader, model)
print ('Done creating trainer')

trainer.train()


