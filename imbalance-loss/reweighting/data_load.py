# -*- coding: utf-8 -*-
#/usr/bin/python3
import tensorflow as tf
from utils import calc_num_batches
from hparams import Hparams

def load_vocab(vocab_fpath):
    vocab = [line.split()[0] for line in open(vocab_fpath, 'r').read().splitlines()]
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    return token2idx, idx2token

def load_data(fpath1, fpath2, maxlen1, maxlen2):
    sents1, sents2 = [], []
    cnt = 0
    with open(fpath1, 'r') as f1, open(fpath2, 'r') as f2:
        for sent1, sent2 in zip(f1, f2):
            if len(sent1.split()) + 1 > maxlen1: continue # 1: </s>
            if len(sent2.split()) + 1 > maxlen2: continue  # 1: </s>
            cnt += 1
            sents1.append(sent1.strip())
            sents2.append(sent2.strip())
    print ('load_data cnt: ', cnt)
    return sents1, sents2


def load_data_eval(fpath1, fpath2, fapth3, maxlen1, maxlen2):
    eval_content = []
    cnt = 0
    with open(fpath1, 'r') as f1, open(fpath2, 'r') as f2, open(fapth3, 'r') as f3:
        for sent1, sent2, sent3 in zip(f1, f2, f3):
            if len(sent1.split()) + 1 > maxlen1: continue # 1: </s>
            if len(sent2.split()) + 1 > maxlen2: continue  # 1: </s>
            eval_content.append(sent3)
    with open('eval3', 'w') as fout:
        fout.write(''.join(eval_content))

def encode(inp, type, dict):

    if Hparams.prefix.startswith('/data'):
        inp_str = inp.decode("utf-8")
    else: # local run don't need encode
        inp_str = inp
    if type=="x": tokens = inp_str.split() + ["</s>"]
    else: tokens = ["<s>"] + inp_str.split() + ["</s>"]

    x = [dict.get(t, dict["<unk>"]) for t in tokens]
    return x

def dprint(x):
    print ('=' * 40)
    (x, x_seqlen, sent1), (decoder_input, y, y_seqlen, sent2, loss_weight) = x
    print ('x: ', x)
    print ('x_seqlen: ', x_seqlen)
    print ('sent1: ', sent1)
    print ('-' * 20)
    print ('decoder_input: ', decoder_input)
    print ('y: ', y)
    print ('y_seqlen: ', y_seqlen)
    print ('sent2: ', sent2)
    print ('loss_weight: ', loss_weight)

def generator_fn(sents1, sents2, vocab_fpath):
    token2idx, _ = load_vocab(vocab_fpath)
    cnt = 0
    for sent1, sent2 in zip(sents1, sents2):
        x = encode(sent1, "x", token2idx)
        y = encode(sent2, "y", token2idx)
        decoder_input, y = y[:-1], y[1:]
        #loss_weight = [0 for i in range(len(y))]
        loss_weight = [((len(y)-0.5-i)/(len(y)*1.0))*0.1 - 0.05 for i in range(len(y))]
        #loss_weight = [((len(y)-i)/(len(y)*1.0))*0.02 - 0.01 for i in range(len(y))]
        x_seqlen, y_seqlen = len(x), len(y)
        if cnt < 10:
            dprint(((x, x_seqlen, sent1), (decoder_input, y, y_seqlen, sent2, loss_weight)))
        cnt += 1
        yield (x, x_seqlen, sent1), (decoder_input, y, y_seqlen, sent2, loss_weight)

def input_fn(sents1, sents2, vocab_fpath, batch_size, shuffle=False):

    shapes = (([None], (), ()),
              ([None], [None], (), (), [None]))
    types = ((tf.int32, tf.int32, tf.string),
             (tf.int32, tf.int32, tf.int32, tf.string, tf.float32))
    paddings = ((0, 0, ''),
                (0, 0, 0, '', 0.))

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2, vocab_fpath))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)
    return dataset

def get_batch(fpath1, fpath2, maxlen1, maxlen2, vocab_fpath, batch_size, shuffle=False):
    sents1, sents2 = load_data(fpath1, fpath2, maxlen1, maxlen2)
    batches = input_fn(sents1, sents2, vocab_fpath, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(sents1), batch_size)
    return batches, num_batches, len(sents1)

if __name__ == '__main__':
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    load_data_eval(hp.eval1, hp.eval2, hp.eval3, hp.maxlen1, hp.maxlen2)
