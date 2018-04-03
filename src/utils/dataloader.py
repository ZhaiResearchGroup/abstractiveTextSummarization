import sys
sys.path.append("..")
from torchtext.data import RawField
from torchtext import data
import torchtext.vocab as vocab
from torchtext.data import RawField
import csv
import os
import pickle
import dill
from utils.sent2vec import Sent2vec
from config.constants import *
import numpy as np
import gensim
import torch
from torch import nn

class Dataloader(object):
    def __init__(self, opt):
        self.cuda = opt.cuda
       #self.batch_size = opt.batch_size
        self.train_data = None 
        self.test_data = None
        self.data_dir = opt.traindata
        self.test_data_dir = opt.testdata
        self.vocab =  None
        self.max_vocab_size = opt.vocab_size
        self.batch_size = opt.batch_size
        self.sent2vec = Sent2vec(opt)
        #self.sent2vec = gensim.models.doc2vec.Doc2Vec.load("/Data/apnews_model/apnews_sen_model.model")
        
        self.load_data()
        
    def load_data(self):
        TEXT = data.Field(sequential=True, lower=True)
        NUM = data.RawField()
        SEN_VEC = data.RawField(postprocessing=self.sen_vec_postprocess)
        SEN_IDX = data.RawField(postprocessing=self.sen_idx_postprocess)
        
        # need to do this so we dont get errors about having too big of a file in a single cell of a csv
        csv.field_size_limit(500 * 1024 * 1024)
        self.train_data, self.test_data = data.TabularDataset.splits( path='',train=self.data_dir, test = self.test_data_dir, format='csv', 
                                            skip_header = True, fields=[('query_num', NUM),
                                                                        ('title', TEXT),
                                                                        ('raw_query', TEXT),
                                                                        ('sum', TEXT),
                                                                        ('story', TEXT),
                                                                        ('sen_vec', SEN_VEC),
                                                                        ('sen_idx', SEN_IDX)])
        
        TEXT.build_vocab(self.train_data, vectors="glove.6B.100d", max_size=self.max_vocab_size)
        
        self.vocab = TEXT.vocab
        
    def get_batch_iterator(self, is_train=True, shuffle=True, repeat=False):
        dataset = self.train_data if is_train  else  self.test_data
        
        dataset_iter = data.BucketIterator(dataset, batch_size=self.batch_size,
                                           device=-1*int(not self.cuda),
                                           sort_key=lambda x: len(x.story),
                                           train=is_train, shuffle=shuffle,
                                           repeat=repeat, sort=(not is_train))
        
        dataset_iter.create_batches()
        
        return dataset_iter
    
    '''
    @args:
        batch: a list of strings (lenght batch_size) where each string is a document 
            to be converted to sen_vec representation
    @returns: 
        torch.Variable that is (batch_size x num_sens_in_doc x 100) 
                This array is the sentences embeddings for each sentence 
                in each document in the batch
    '''
    #@staticmethod
    def sen_vec_postprocess(self, batch):

        # split at <sos> tokens
        tokenized_batch = [[SOS_TOKEN + sen for sen in example.split(SOS_TOKEN)][1:] for example in batch]
        # tokenized batch is now a list[list[sentences]]

        # maximum length of any document (in number of sentences)
        max_doc_len = np.max(np.array([len(ex) for ex in tokenized_batch]))

        batch_vec = np.zeros((len(batch), max_doc_len+1, 100))

        for i, example in enumerate(tokenized_batch):
            # length of this example tells us how much we need to leave as padding on the end
            batch_vec[i,:len(example),:] = self.sent2vec.infer_vector(example)

        # return as cuda var
        tensor = torch.FloatTensor(batch_vec)
        if self.cuda:
            tensor = tensor.cuda()
        return torch.autograd.Variable(tensor)
    
    '''
        @args:
            batch: a list of strings where each string is a document for which 
                we want to generate a mapping from word indcies to sentence indcies 
                    (same input as sen_vec_postprocess)
        @returns: 
            torch.Variable that is (batch_size x num_sens_in_doc x 100) 
                    This array is the sentences embeddings for each sentence 
                    in each document in the batch
    '''
    #@staticmethod
    def sen_idx_postprocess(self, batch):
        # split at <sos> tokens
        sentences_batch = [[SOS_TOKEN + sen for sen in example.split(SOS_TOKEN)][1:] for example in batch]
        # tokenized batch is now a list[list[sentences]]
        max_num_sens = max([len(ex) for ex in sentences_batch])

        # words batch is now a list[list[words]]
        words_batch = [example.split() for example in batch]
        # maximum length of any document (in number of words)
        max_doc_len = max([len(ex) for ex in words_batch])


        # initialize the array of sentence indices
        # we use one more than the maximum to be the zero vector.
        # In sentence attentions, this always receives zero weight
        sen_idxs = np.ones((len(batch), max_doc_len)) * max_num_sens

        # TODO: can someone check this and make sure it makes sense??
        for sen_idx, example in enumerate(sentences_batch):
            curr = 0
            for wd_idx, sent in enumerate(example):
                sen_idxs[sen_idx, curr:curr+len(sent.split())] = wd_idx
                curr += len(sent.split())

        if self.cuda:
            return torch.autograd.Variable(torch.LongTensor(sen_idxs).cuda())
        else:
            return torch.autograd.Variable(torch.LongTensor(sen_idxs))
