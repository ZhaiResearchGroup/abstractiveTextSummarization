import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torchtext.vocab as vocab
import numpy as np
import argparse

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
USE_CUDA = True

word_dim      = 100
input_length  = 50
latent_dim    = 30
learning_rate = 0.001
training_iterations = 10000
print_every = 1000

class senEncoder(nn.Module):
    def __init__(self, input_length, word_dim, latent_dim):
        super(senEncoder, self).__init__()

        self.input_length = input_length
        self.word_dim = word_dim
        self.latent_dim = latent_dim

        self.s2filter = nn.Conv2d(1, latent_dim, (2, word_dim))
        self.s3filter = nn.Conv2d(1, latent_dim, (3, word_dim))
        self.s4filter = nn.Conv2d(1, latent_dim, (4, word_dim))

        self.s2pool = nn.MaxPool1d(input_length - 1, return_indices=True)
        self.s3pool = nn.MaxPool1d(input_length - 2, return_indices=True)
        self.s4pool = nn.MaxPool1d(input_length - 3, return_indices=True)

    def forward(self, sent_matrix):
        sent_matrix = sent_matrix.unsqueeze(1)

        print(sent_matrix.shape)

        s2features = self.s2filter(sent_matrix).squeeze(3)
        s3features = self.s3filter(sent_matrix).squeeze(3)
        s4features = self.s4filter(sent_matrix).squeeze(3)
        print('s2 features size: ', s2features.shape)
        print('s3 features size: ', s3features.shape)
        print('s4 features size: ', s4features.shape)

        s2pooled, s2indices = self.s2pool(s2features)
        s3pooled, s3indices = self.s3pool(s3features)
        s4pooled, s4indices = self.s4pool(s4features)
        print('s2 features size: ', s2pooled.shape)
        print('s3 features size: ', s3pooled.shape)
        print('s4 features size: ', s4pooled.shape)

        latent = torch.cat((s2pooled, s3pooled, s4pooled), 1)
        indices = torch.cat((s2indices, s3indices, s4indices), 1)
        return latent, indices

class senDecoder(nn.Module):
    def __init__(self, input_length, word_dim, latent_dim):
        super(senDecoder, self).__init__()

        self.input_length = input_length
        self.word_dim = word_dim
        self.latent_dim = latent_dim

        self.s2unpool = nn.MaxUnpool1d(input_length - 1)
        self.s3unpool = nn.MaxUnpool1d(input_length - 2)
        self.s4unpool = nn.MaxUnpool1d(input_length - 3)

        self.s2deconv = nn.ConvTranspose2d(latent_dim, 1, (2, word_dim))
        self.s3deconv = nn.ConvTranspose2d(latent_dim, 1, (3, word_dim))
        self.s4deconv = nn.ConvTranspose2d(latent_dim, 1, (4, word_dim))

    def forward(self, latent, indices):
        print(indices[:,self.latent_dim * 2:,:].shape)
        print(latent[:,self.latent_dim * 2:,:].shape)
        lat = self.latent_dim
        s2unpooled = self.s2unpool(latent[:,:lat,:],
                                   indices[:,:lat,:]).unsqueeze(3)
        s3unpooled = self.s3unpool(latent[:,lat:lat * 2,:],
                                   indices[:,lat:lat * 2,:]).unsqueeze(3)
        s4unpooled = self.s4unpool(latent[:,lat * 2:,:],
                                   indices[:,lat * 2:,:]).unsqueeze(3)
        print('s2 unpooled size: ', s2unpooled.shape)
        print('s3 unpooled size: ', s3unpooled.shape)
        print('s4 unpooled size: ', s4unpooled.shape)

        s2reconst = self.s2deconv(s2unpooled)
        s3reconst = self.s3deconv(s3unpooled)
        s4reconst = self.s4deconv(s4unpooled)

        print('s2reconst size: ', s2reconst.shape)
        print('s3reconst size: ', s3reconst.shape)
        print('s4reconst size: ', s4reconst.shape)

        res = s2reconst + s3reconst + s4reconst
        return res

# load GloVe vectors from torchtext
glove = vocab.GloVe(name='6B', dim=100)
print('Loaded {} words'.format(len(glove.itos)))

def glove_encode(batch, word_dim, input_length):
    batch_tok = [sent.split() for sent in batch]
    sent_lengths = [len(sent) for sent in batch_tok]
    batch_vectors = np.zeros((len(batch_tok), input_length, word_dim))
    for sent_idx in range(len(batch_tok)):
        words = batch_tok[sent_idx]
        sent_vectors = [glove.vectors[glove.stoi[w]] for w in words]
        sent_mat = np.stack(sent_vectors, axis=0)
        batch_vectors[sent_idx,:sent_lengths[sent_idx]] = sent_mat
    return batch_vectors

def train(input_batch, encoder, decoder, enc_optim, dec_optim, criterion):
    enc_optim.zero_grad()
    dec_optim.zero_grad()

    input_mat = glove_encode(input_batch, word_dim, input_length)
    print('Input mat: ', input_mat.shape)
    input_var = Variable(torch.FloatTensor(input_mat))
    target_var = Variable(torch.FloatTensor(input_mat))

    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    latent, indices = encoder(input_var)
    reconst = decoder(latent, indices)

    # compute and propagate loss gradient
    loss = criterion(reconst, target_var)
    loss.backward()

    enc_optim.step()
    dec_optim.step()

    return loss

class Sent2vec():
    def __init__(self, opt, eval):
        self.cuda = opt.cuda
        self.train_data = None
         

    def train
if __name__ == "__main__":
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
           
    data = Dataloader(opt, False)

    # set up encoder and decoder
    encoder = senEncoder(input_length, word_dim, latent_dim)
    decoder = senDecoder(input_length, word_dim, latent_dim)
    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

# run training iterations

batch = ['the curtain lifts to a hushed crowd',
         'all the world is a stage',
         'and each man must play his own part']
train(batch, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
