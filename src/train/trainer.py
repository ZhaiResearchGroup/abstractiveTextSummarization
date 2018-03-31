import sys
sys.path.append("..")
from utils import utils
from utils.dataloader import Dataloader
import torch
from torch import nn
from torch import optim
import numpy as np
from utils.criterion import masked_cross_entropy
from config.constants import *


class Trainer(object):
    def __init__(self, opt, dataloader, model):
        super(Trainer, self).__init__()
        
        self.loader = dataloader
        self.model = model
        self.learning_rate = opt.learning_rate
        self.n_epochs = opt.n_epochs
        self.use_cuda = opt.cuda
        
        if hasattr(opt, 'optimizer_type'):
            self.optimizer_type = opt.optimizer_type
        else:
            self.optimizer_type = optim.Adam
            
        if hasattr(opt, 'clip'):
            self.clip = opt.clip
        else:
            self.clip = 50.0
    
    def train(self):
        # build optimizers using parameters and learning ratios from model
        optimizers = [self.optimizer_type(parameters, self.learning_rate*ratio) for parameters, ratio in self.model.get_parameters()]
        all_parameters = [parameters for parameters, ratio in self.model.get_parameters()]
        
        # initialize clip values and loss
        param_clips = np.zeros(len(all_parameters))
        total_loss = 0
        
        # loop over number of epochs
        for epoch in range(self.n_epochs):
            
            # iterate over batches
            for batch in self.loader.get_batch_iterator(is_train=True):
                # zero out optimizers
                for optimizer in optimizers:
                    optimizer.zero_grad()
                
                # parse batch
                input_batch, max_input_length, tgt_batch, max_tgt_length, sen_vecs, sen_idxs = utils.parse_batch(batch, use_cuda=self.use_cuda)
                                
                # run forward pass over model
                model_output = self.model(input_batch, max_input_length, tgt_batch, max_tgt_length, sen_vecs, sen_idxs)
                
                # get loss
                loss = masked_cross_entropy(model_output.transpose(0, 1).contiguous(), # -> batch x seq
                                            tgt_batch.transpose(0, 1).contiguous(), # -> batch x seq
                                            #max_tgt_length,
                                            torch.LongTensor([self.loader.vocab.stoi[SOS_TOKEN]] * tgt_batch.size(1)),
                                            use_cuda=self.use_cuda)
                
                # run backward pass over graph
                loss.backward()
                
                # update total loss
                total_loss += loss.data[0]
                
                # Clip gradient norms
                for i, parameters in enumerate(all_parameters):
                    param_clips[i] += nn.utils.clip_grad_norm(parameters, self.clip).data[0]
                
                # update parameters
                for optimizer in optimzers:
                    optimizer.step()
                    
                print ('loss:', loss.data[0])
                    
                
