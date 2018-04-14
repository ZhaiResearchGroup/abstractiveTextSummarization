import sys
sys.path.append("..")
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_vectors, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding.weight.data.copy_(vocab_vectors)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        
    # input seqs should be (batch_size, batch_len)
    def forward(self, input_seqs, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)

        outputs, hidden = self.gru(embedded, hidden)

        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        
        #hidden.detach_() ####
        # outputs are shape (batch_size, hidden_size)
        return outputs, hidden