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

input_length  = 80
word_dim      = 100
latent_dim    = (30, 30, 40)
learning_rate = 0.001
training_iterations = 10000
print_every = 1000
vocab_size = 400000

class senEncoder(nn.Module):
    def __init__(self, input_length, word_dim, latent_dim):
        super(senEncoder, self).__init__()

        self.input_length = input_length
        self.word_dim = word_dim
        self.latent_dim = latent_dim

        self.s2filter = nn.Conv2d(1, latent_dim[0], (2, word_dim))
        self.s3filter = nn.Conv2d(1, latent_dim[1], (3, word_dim))
        self.s4filter = nn.Conv2d(1, latent_dim[2], (4, word_dim))

        self.s2pool = nn.MaxPool1d(input_length - 1, return_indices=True)
        self.s3pool = nn.MaxPool1d(input_length - 2, return_indices=True)
        self.s4pool = nn.MaxPool1d(input_length - 3, return_indices=True)

    def forward(self, sent_matrix):
        sent_matrix = sent_matrix.unsqueeze(1)

        s2features = self.s2filter(sent_matrix).squeeze(3)
        s3features = self.s3filter(sent_matrix).squeeze(3)
        s4features = self.s4filter(sent_matrix).squeeze(3)

        s2pooled, s2indices = self.s2pool(s2features)
        s3pooled, s3indices = self.s3pool(s3features)
        s4pooled, s4indices = self.s4pool(s4features)

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

        self.s2deconv = nn.ConvTranspose2d(latent_dim[0], 1, (2, word_dim))
        self.s3deconv = nn.ConvTranspose2d(latent_dim[1], 1, (3, word_dim))
        self.s4deconv = nn.ConvTranspose2d(latent_dim[2], 1, (4, word_dim))

    def forward(self, latent, indices):
        lat = self.latent_dim
        s2unpooled = self.s2unpool(latent[:,:lat[0],:],
                                   indices[:,:lat[0],:]).unsqueeze(3)
        s3unpooled = self.s3unpool(latent[:,lat[0]:lat[1],:],
                                   indices[:,lat[0]:lat[1],:]).unsqueeze(3)
        s4unpooled = self.s4unpool(latent[:,lat[1]:,:],
                                   indices[:,lat[1]:,:]).unsqueeze(3)

        s2reconst = self.s2deconv(s2unpooled)
        s3reconst = self.s3deconv(s3unpooled)
        s4reconst = self.s4deconv(s4unpooled)

        res = s2reconst + s3reconst + s4reconst
        return res

class Sent2vec(object):
    def __init__(self, opt):
        self.glove = vocab.GloVe(name='6B', dim=100)
        self.glove.itos.append(opt.sos)
        self.glove.stoi[opt.sos]=vocab_size
        self.glove.vectors = torch.cat([self.glove.vectors, torch.zeros(1, word_dim)], 0)
        print('Loaded {} words'.format(len(self.glove.itos)))
        self.enc_path = os.path.join(opt.load_dir, 'enc_model.sav')
        self.dec_path = os.path.join(opt.load_dir, 'dec_model.sav')
        if os.path.isfile(self.enc_path) and os.path.isfile(self.dec_path):
            self.encoder = torch.load(self.enc_path)
            self.decoder = torch.load(self.dec_path)
        else:
            self.encoder = senEncoder(input_length, word_dim, latent_dim)
            self.decoder = senDecoder(input_length, word_dim, latent_dim)
        if USE_CUDA:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
        if 'train_ae' in opt:
            print('Autoencoder is set up to train.')
            self.enc_optim = optim.Adam(self.encoder.parameters(), lr=learning_rate)
            self.dec_optim = optim.Adam(self.decoder.parameters(), lr=learning_rate)
            self.criterion = nn.MSELoss()

    def train(self, input_var):
        #print(batch)
        # input_var = Variable(torch.FloatTensor(batch))
        # target_var = Variable(torch.FloatTensor(batch))
        # if USE_CUDA:
        #     input_var = input_var.cuda()
        #     target_var = target_var.cuda()
        print(input_var.size())
        latent, indices = self.encoder(input_var)
        reconst = self.decoder(latent, indices)
        # compute and propagate loss gradient
        loss = self.criterion(reconst, input_var)
        loss.backward()
        self.enc_optim.step()
        self.dec_optim.step()

    def save(self):
        self.enc_path = os.path.join(opt.load_dir, 'enc_model.sav')
        self.dec_path = os.path.join(opt.load_dir, 'dec_model.sav')
        torch.save(self.encoder, self.enc_path)
        torch.save(self.decoder, self.dec_path)

    def infer_vector(self, batch):
        #print(batch)
        sent_matrix = self.__glove_encode(batch, word_dim, input_length)
        #print(sent_matrix.shape)
        input_var = Variable(torch.FloatTensor(sent_matrix))
        if USE_CUDA:
            input_var = input_var.cuda() 
        latent_vector,_ = self.encoder(input_var)
        return latent_vector.data.squeeze(2)

    def __glove_encode(self, batch, word_dim, input_length):
        batch_tok = [sent.split() for sent in batch]
        sent_lengths = [len(sent) for sent in batch_tok]
        batch_vectors = np.zeros((len(batch_tok), input_length, word_dim))
        for sent_idx in range(len(batch_tok)):
            words = batch_tok[sent_idx]
            sent_mat = np.zeros((len(words), word_dim))
            for w_idx, w in enumerate(words):
                if w in self.glove.stoi:
                    sent_mat[w_idx] = self.glove.vectors[self.glove.stoi[w]]
                else:
                    sent_mat[w_idx] = np.zeros(word_dim)
            batch_vectors[sent_idx,:sent_lengths[sent_idx]] = sent_mat[:input_length]
        return batch_vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train.py')
    ## Data options
    parser.add_argument('-trnd', '--traindata', default='../../data/wiki_queries12_head.csv', help='Path to train data file')
    parser.add_argument('-tstd', '--testdata', default='../../data/wiki_queries12_head.csv', help="Path to the test data file")
    parser.add_argument("-processeddata", default='dataset/data.pkl',help="Path to the pre-processed data set")
    parser.add_argument("-sos",default="<sos>",help='Adding EOS token at the end of each sequence')
    parser.add_argument('-sdir', '--save_dir', default='saving', help='Directory to save model checkpoints')
    parser.add_argument('-ldir', '--load_dir', default='loading', help='Path to a model checkpoint')
    parser.add_argument("-vocab_size", type=int, default=None, help="Limit vocabulary")
    parser.add_argument('-batch_size', type=int, default=32, help='Batch Size for seq2seq model')

    parser.add_argument('-gpu', type=int, default=-1,help='GPU id. Support single GPU only')

    opt = parser.parse_args()
    opt.cuda = (opt.gpu != -1)
    opt.train_ae = True

    s2v = Sent2vec(opt)
    from dataloader import Dataloader
    data = Dataloader(opt)
    b_iter = dataloader.get_batch_iterator()
    for batch in b_iter:
        s2v.train(batch)
