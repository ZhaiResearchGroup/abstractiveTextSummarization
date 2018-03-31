from ..utils import utils
from ..utils.dataload import Dataloader
import torch
from torch import nn


def Trainor(object):
    def __init__(self, opt):
        self.loader = Dataloader(opt)
        self.model = opt.model
        self.n_epochs = opt.n_epochs
        self.use_cuda = opt.cuda
    
    def train(self):
        # loop over number of epochs
        for epoch in range(self.n_epochs):
            
            # iterate over batches
            for batch in loader.get_batch_iterator(is_train=True):
                decoder_output = self.model(*utils.parse_batch(batch, use_cuda=self.use_cuda))
                
                loss = masked_cross_entropy(decoder_output.transpose(0, 1).contiguous(), # -> batch x seq
                                            target_batches.transpose(0, 1).contiguous(), # -> batch x seq
                                            target_lengths,
                                           )
                