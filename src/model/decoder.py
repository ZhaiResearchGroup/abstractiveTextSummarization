import sys
sys.path.append("..")
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.attn import Attn
from config.constants import *


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_vectors, query_length=50, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        
        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
                
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding.weight.data.copy_(vocab_vectors)

        self.dropout = nn.Dropout(dropout_p)
        self.word_attn = Attn()
        self.sentence_attn = Attn() # set at start of notebook
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)
        
        self.first_iter = True
    
    def forward(self, input_seq, last_hidden, encoder_outputs, sentence_vectors, sentence_idx, sentence_attn_weights, query_batch):
        # Get the embedding of the current input word (last output word)
        embedded = self.embedding(input_seq)
        embedded = self.dropout(embedded)
        embedded = embedded.unsqueeze(0)
        
        if self.first_iter:
            sentence_attn_weights = self.sentence_attn(query_batch, sentence_vectors.transpose(0,1).transpose(1,2)).transpose(0,1)
            self.first_iter = False
                    
        # wtf are dimensions? (changed so that batch is the middle dimension)
        word_attn_weights = self.word_attn(last_hidden[-1].unsqueeze(2).transpose(1,2), encoder_outputs.transpose(0, 1).transpose(1,2)).transpose(0,1)

        indices_var = sentence_idx.transpose(1,0).unsqueeze(0)
        
        stretched_sent_attn_weights = torch.gather(sentence_attn_weights, 2, indices_var)

        attn_weights = F.softmax(word_attn_weights * stretched_sent_attn_weights, dim=2)
        
        context = attn_weights.transpose(0,1).bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        context = context.transpose(0, 1) # 1 x B x N
        
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((embedded, context), 2)
        output, hidden = self.gru(rnn_input, last_hidden)

        # Final output layer
        output = output.squeeze(0) # B x N
        context = context.squeeze(0) # B x N
        output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        
        return output, hidden, sentence_attn_weights
    
    def _detach_hidden(self, hidden):
        if type(hidden) == Variable:
                hidden.detach_() # same as creating a new variable.
        else:
            for h in hidden: h.detach_() 