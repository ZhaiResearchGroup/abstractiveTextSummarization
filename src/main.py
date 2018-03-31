import argparse
import os
import random
import torch
import torch.nn as nn
import numpy as np
from torch import cuda
from utils.dataset import Dataset
parser = argparse.ArgumentParser(description='train.py')
## Data options
parser.add_argument('-traindata', default='dataset/wiki_short.csv', help='Path to train data file')
parser.add_argument("-testdata", default='dataset/wiki_short.csv',help="Path to the test data file")
parser.add_argument("-processeddata", default='dataset/data.pkl',help="Path to the pre-processed data set")
parser.add_argument("-sos",default="<sos>",help='Adding EOS token at the end of each sequence')

parser.add_argument('-save_dir',default='saving', help='Directory to save model checkpoints')
parser.add_argument('-load_from', type=str, help='Path to a model checkpoint')
parser.add_argument("-vocab_size", type=int, default=None, help="Limit vocabulary")
parser.add_argument('-batch_size', type=int, default=32, help='Batch Size for seq2seq model')

parser.add_argument('-gpu', type=int, default=-1,help='GPU id. Support single GPU only')

opt = parser.parse_args()
opt.cuda = (opt.gpu != -1)

dataset = Dataset(opt, False)
print(type(dataset.train_data))