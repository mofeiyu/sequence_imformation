# -*- coding: utf-8 -*-
#/usr/bin/python2

import os

class Hyperparams:
    '''Hyperparameters'''
    # data
    
    prefix = '/data/mt/'
    if not os.path.exists(prefix):
        prefix = '../data/'
    source_train = prefix + 'de-en/train.tags.de-en.de'
    target_train = prefix + 'de-en/train.tags.de-en.en'
    source_test = prefix + 'de-en/IWSLT16.TED.tst2014.de-en.de.xml'
    target_test = prefix + 'de-en/IWSLT16.TED.tst2014.de-en.en.xml'
    de_vocab = prefix + 'preprocessed/de.vocab.tsv'
    en_vocab = prefix + 'preprocessed/en.vocab.tsv'
    
    # training
    batch_size = 32 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir' # log directory
    
    # model
    maxlen = 15 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 5 # words whose occurred less than min_cnt are encoded as <UNK>.
    emb_dim = 128 # alias = C
    
    rnn_size = 128
    rnn_num_layers = 1
    rnn_beam_width = 5
    rnn_type = 'lstm'
    
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 30
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    

