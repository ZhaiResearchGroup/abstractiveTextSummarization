import sys
sys.path.append("..")
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from utils.utils import parse_batch
from model.encoder import EncoderRNN
from model.decoder import BahdanauAttnDecoderRNN
from config.constants import *

class Seq2Seq(nn.Module):
    def __init__(self,  opt, vocab):
        super(Seq2Seq, self).__init__()
        
        self.word_size = len(vocab)
        
        ## CHANGE THIS TO CREATE THE ENOCDER AND DECODER HERE ###
        self.encoder = EncoderRNN(self.word_size, 100, vocab.vectors)
        self.decoder = BahdanauAttnDecoderRNN(100, self.word_size, vocab.vectors)
        self.vocab = vocab
        self.opt = opt
        
        if hasattr(opt, 'decoder_learning_ratio'):
            self.decoder_learning_ratio = opt.decoder_learning_ratio
        else:
             self.decoder_learning_ratio = 5.0   
        
        # # Share the embedding matrix - preprocess with share_vocab required.
        # if opt.share_embeddings:
        #     self.encoder.embedding.weight = self.decoder.embedding.weight


    def _initialize_forward(self, input_batch, max_tgt_len, sen_vecs, eval):

        batch_size = input_batch.size(1)

        # Decoder's input
        init_decoder_input_seq = Variable(torch.LongTensor([self.vocab.stoi[SOS_TOKEN]] * batch_size),volatile=eval)

        # Var to Store all decoder's outputs.
        # **CRUTIAL**
        # Don't set:
        # >> decoder_outputs = Variable(torch.zeros(max_tgt_len, batch_size, decoder.vocab_size))
        # Varying tensor size could cause GPU allocate a new memory causing OOM,
        # so we intialize tensor with fixed size instead:
        # opts.max_seq_len is a fixed number, unlike `max_tgt_len` always varys.
        
        # TODO find max tgt length (instead of 1000 or max_tgt_len)
        decoder_outputs = Variable(torch.zeros(max_tgt_len, batch_size, self.word_size))
        
        sentence_attn_weights = Variable(torch.zeros(1, batch_size, sen_vecs.shape[0]))

        # Move variables from CPU to GPU.
        if self.opt.cuda:
            init_decoder_input_seq = init_decoder_input_seq.cuda()
            decoder_outputs = decoder_outputs.cuda()
            sentence_attn_weights = sentence_attn_weights.cuda()

        # -------------------------------------
        # Forward encoder
        # -------------------------------------
        encoder_outputs, encoder_hidden = self.encoder(input_batch)

        
        # -------------------------------------
        # Forward decoder
        # -------------------------------------
        # Initialize decoder's hidden state as encoder's last hidden state.
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        
        return encoder_outputs, init_decoder_input_seq, decoder_outputs, decoder_hidden, sentence_attn_weights


    def forward(self, input_batch, max_input_length, tgt_batch, max_tgt_len, sen_vecs, sen_idxs, eval=False, regression=False):
        
        encoder_outputs, decoder_input, decoder_outputs, decoder_hidden, sen_attention_weights = self._initialize_forward(input_batch, max_tgt_len, sen_vecs, eval)

        if eval:
            use_teacher_forcing = False
        else:
            use_teacher_forcing = (random.random() < self.opt.teacher_forcing_ratio)
            
        # Run through decoder one time step at a time.
        for t in range(max_tgt_len):
            # decoder returns:
            # - decoder_output   : (batch_size, vocab_size)
            # - decoder_hidden   : (num_layers, batch_size, hidden_size)
            # - sen_attention_weights: (batch_size, num_sens)
            decoder_output, decoder_hidden, sen_attention_weights = self.decoder(decoder_input, decoder_hidden, encoder_outputs, sen_vecs, sen_idxs, sen_attention_weights)

            # Store decoder outputs.
            decoder_outputs[t] = decoder_output
            #print type(tgt_seqs[t])
            if use_teacher_forcing:
                # Next input is current target
                #print 'tf'
                decoder_input = tgt_batch[t]
            else:
                #print 'not tf'
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze()
                # print(type (input_seq.detach()))
                # print input
                if self.opt.cuda:
                    decoder_input = decoder_input.cuda()
    
        # why?  maybe to make sure it doesnt get deleted?
        self.decoder_outputs = decoder_outputs
        if regression:
            self.decoder_outputs = self.decoder_outputs.view_as(tgt_batch)
        return self.decoder_outputs
    
    def get_parameters(self):
        return [(self.encoder.parameters(), 1), (self.decoder.parameters(), self.decoder_learning_ratio)]

    def translate(self, inputs):
        src_seqs, src_lens = inputs[0]
        max_tgt_len, encoder_outputs, src_lens, tgt_lens, decoder_outputs, tgt_seqs, \
        input_seq, decoder_hidden = self.initialize(inputs, eval=True)

        if self.opt.attention:
            all_attention_weights = torch.zeros(self.opt.max_train_decode_len, src_seqs.size(1), len(tgt_seqs))
        # Run through decoder one time step at a time.
        end_of_batch_pred = np.array([lib.Constants.EOS] * len(src_lens))
        preds = np.ones((self.opt.max_train_decode_len, len(src_lens))) * 2
        for t in range(self.opt.max_train_decode_len):

            # decoder returns:
            # - decoder_output   : (batch_size, vocab_size)
            # - decoder_hidden   : (num_layers, batch_size, hidden_size)
            # - attention_weights: (batch_size, max_src_len)
            decoder_output, decoder_hidden, attention_weights = self.decoder(input_seq, decoder_hidden,
                                                                        encoder_outputs, src_lens)

            if self.opt.attention:
                # Store attention weights.
                all_attention_weights[t] = attention_weights.cpu().data
            # Choose top word from decoder's output
            prob, token_ids = decoder_output.data.topk(1)
            #print token_ids
            #print (prob, token_ids)

            # Next input is chosen word
            token_ids = token_ids.squeeze()
            preds[t,:] = token_ids
            input_seq = Variable(token_ids, volatile=True)
            if self.opt.cuda: input_seq = input_seq.cuda()

            if np.sum(np.equal(token_ids.cpu().numpy(),end_of_batch_pred)) == len(src_seqs):
                #out_sent.append(tokens)
                #out_tids.append(token_ids)
                break
                # this never hits...
                assert False
            # Repackage hidden state (may not need this, since no BPTT)
            #self.detach_hidden(decoder_hidden)
        preds = torch.LongTensor(preds)
        return preds

    def detach_hidden(self, hidden):
        """ Wraps hidden states in new Variables, to detach them from their history. Prevent OOM.
            After detach, the hidden's requires_grad=Fasle and grad_fn=None.
        Issues:
        - Memory leak problem in LSTM and RNN: https://github.com/pytorch/pytorch/issues/2198
        - https://github.com/pytorch/examples/blob/master/word_language_model/main.py
        - https://discuss.pytorch.org/t/help-clarifying-repackage-hidden-in-word-language-model/226
        - https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426
        -
        """
        if type(hidden) == Variable:
            hidden.detach_() # same as creating a new variable.
        else:
            for h in hidden: h.detach_()

    def sample(self, inputs, max_length):
        max_tgt_len, encoder_outputs, src_lens, tgt_lens, decoder_outputs, tgt_seqs, \
        input_seq, decoder_hidden = self.initialize(inputs, eval=False)

        outputs = []
        samples = []
        batch_size = tgt_seqs.size(1)
        num_eos = tgt_seqs[0].data.byte().new(batch_size).zero_()

        for t in range(max_length):
            decoder_output, decoder_hidden, attention_weights = self.decoder(input_seq, decoder_hidden, encoder_outputs, src_lens)
            outputs.append(decoder_output)
            dist = F.softmax(decoder_output, dim=-1)
            sample = dist.multinomial(1, replacement=False).view(-1).data
            samples.append(sample)

            # Stop if all sentences reach EOS.
            num_eos |= (sample ==  lib.Constants.EOS)
            if num_eos.sum() == batch_size: break

            # Next input is chosen word
            input_seq = Variable(sample,volatile=False)

        outputs = torch.stack(outputs)
        samples = torch.stack(samples)
        return samples, outputs

    def index_select_decoder_state(self, pos):
        self.decoder_outputs = self.decoder_outputs.index_select(0, pos)
        self.decoder.index_select_state(pos)

    def reset(self, batch_size, eval=False):
        self.torch = torch.cuda if self.opt.cuda else torch
        self.encoder.hidden = None
        self.decoder.decoder_hidden=None
        self.decoder_outputs = Variable(self.torch.FloatTensor(
            batch_size, self.opt.dec_rnn_size).zero_(), volatile=eval)