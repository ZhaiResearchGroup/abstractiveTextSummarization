import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..utils.dataset import parse_batch

class Seq2Seq(nn.Module):
    def __init__(self,  opt):
        super(Seq2Seq, self).__init__()
        
        ## CHANGE THIS TO CREATE THE ENOCDER AND DECODER HERE ###
        self.encoder = encoder
        self.decoder = decoder
        self.opt = opt
        
        # Share the embedding matrix - preprocess with share_vocab required.
        if opt.share_embeddings:
            self.encoder.embedding.weight = self.decoder.embedding.weight


    def _initialize_forward(self, inputs, eval):
        src_seqs, src_lens = inputs[0]
        tgt_seqs, tgt_lens = inputs[1]

        batch_size = src_seqs.size(1)
        assert(batch_size == tgt_seqs.size(1))
        # Pack tensors to variables for neural network inputs (in order to autograd)
        #src_seqs = Variable(src_seqs)
        #tgt_seqs = Variable(tgt_seqs)

        # Decoder's input
        input_seq = Variable(torch.LongTensor([lib.Constants.BOS] * batch_size),volatile=eval)

        # Decoder's output sequence length = max target sequence length of current batch
        max_tgt_len = tgt_seqs.size()[0]#tgt_lens.data.max()

        # Store all decoder's outputs.
        # **CRUTIAL**
        # Don't set:
        # >> decoder_outputs = Variable(torch.zeros(max_tgt_len, batch_size, decoder.vocab_size))
        # Varying tensor size could cause GPU allocate a new memory causing OOM,
        # so we intialize tensor with fixed size instead:
        # opts.max_seq_len is a fixed number, unlike `max_tgt_len` always varys.
        self.decoder_outputs = Variable(torch.zeros(self.opt.max_train_decode_len, batch_size, self.decoder.out_size))

        # Move variables from CPU to GPU.
        if self.opt.cuda:
            input_seq = input_seq.cuda()
            self.decoder_outputs = self.decoder_outputs.cuda()

        # -------------------------------------
        # Forward encoder
        # -------------------------------------
        encoder_outputs, encoder_hidden = self.encoder(src_seqs, src_lens.data.tolist())

        # -------------------------------------
        # Forward decoder
        # -------------------------------------
        # Initialize decoder's hidden state as encoder's last hidden state.
        decoder_hidden = encoder_hidden

        return max_tgt_len, encoder_outputs, src_lens, tgt_lens, self.decoder_outputs, tgt_seqs, input_seq, decoder_hidden

    def forward(self, inputs, eval=False, regression=False):
        max_tgt_len, encoder_outputs, src_lens, tgt_lens, decoder_outputs, tgt_seqs, \
        input_seq, decoder_hidden = self._initialize_forward(inputs, eval)

        if eval:
            use_teacher_forcing = False
        else:
            use_teacher_forcing = random.random() < self.opt.teacher_forcing_ratio
            
        # Run through decoder one time step at a time.
        for t in range(max_tgt_len):
            # decoder returns:
            # - decoder_output   : (batch_size, vocab_size)
            # - decoder_hidden   : (num_layers, batch_size, hidden_size)
            # - attention_weights: (batch_size, max_src_len)
            decoder_output, decoder_hidden, attention_weights = self.decoder(input_seq, decoder_hidden,encoder_outputs, src_lens)

            # Store decoder outputs.
            decoder_outputs[t] = decoder_output
            #print type(tgt_seqs[t])
            if use_teacher_forcing:
                # Next input is current target
                #print 'tf'
                input_seq = tgt_seqs[t]
            else:
                #print 'not tf'
                topv, topi = decoder_output.topk(1)
                input_seq = topi.squeeze()
                # print(type (input_seq.detach()))
                # print input
                if self.opt.cuda:
                    input_seq = input_seq.cuda()
            # print input_seq.size()
            # Detach hidden state:
            #self.detach_hidden(decoder_hidden)

        self.decoder_outputs = decoder_outputs
        if regression:
            self.decoder_outputs = self.decoder_outputs.view_as(tgt_seqs)
        return self.decoder_outputs


    def backward(self, outputs, tgt_seqs, mask, criterion, eval=False, regression=False, normalize=True):

        max_tgt_len = tgt_seqs.size()[0] #tgt_seqs.size()[0]#tgt_lens.data.max()
        # -------------------------------------
        # Compute loss
        # -------------------------------------
        if regression:
            logits = outputs
            loss = criterion(logits, tgt_seqs, mask, normalize=normalize)
            num_corrects = None
        else:
            logits = outputs[:max_tgt_len]
            loss, num_corrects = criterion(logits, tgt_seqs, mask, normalize=normalize)

            """logits_flat = logits.contiguous().view(-1, logits.size(-1))
            targets_flat  = tgt_seqs.contiguous().view(-1,)
            loss = criterion(logits_flat, targets_flat)
            loss = loss.view(*tgt_seqs.size())
        num_corrects = lib.metric.compute_numcorrects(logits, tgt_seqs, mask)
        loss = loss * mask.float()
        loss = loss.sum()
        loss = loss / mask.float().sum() if normalize else loss"""
        # -------------------------------------
        # Backward and optimize
        # -------------------------------------
        # Backward to get gradients w.r.t parameters in model.
        if(not eval):
            loss.backward()

        # Clip gradients
        #encoder_grad_norm = nn.utils.clip_grad_norm(encoder.parameters(), opts.max_grad_norm)
        #decoder_grad_norm = nn.utils.clip_grad_norm(decoder.parameters(), opts.max_grad_norm)
        #clipped_encoder_grad_norm = compute_grad_norm(encoder.parameters())
        #clipped_decoder_grad_norm = compute_grad_norm(decoder.parameters())

        return loss.data[0], num_corrects

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