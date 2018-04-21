# script to train Sent2Vec CNN autoencoder model

from utils.sent2vec import Sent2vec
from utils.dataloader import Dataloader
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sent2vec_train.py')
    ## Data options
    parser.add_argument('-trnd', '--traindata', default='../data/wiki_queries12_head.csv', help='Path to train data file')
    parser.add_argument('-tstd', '--testdata', default='../data/wiki_queries12_head.csv', help="Path to the test data file")
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
    data = Dataloader(opt)
    b_iter = data.get_batch_iterator()

    try:
        for batch in b_iter:
            s2v.train(batch.story)
        s2v.save()
    except KeyboardInterrupt:
        s2v.save()
