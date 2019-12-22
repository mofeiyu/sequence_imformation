# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function
from hyperparams import Hyperparams as hp
import codecs
import regex
from collections import Counter

def make_vocab(fpath, fname):
    text = codecs.open(fpath, 'r', 'utf-8').read()
    text = regex.sub("[^\s\p{Latin}']", "", text)
    words = text.split()
    word2cnt = Counter(words)
    with codecs.open(fname, 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))

if __name__ == '__main__':
    make_vocab(hp.source_train, hp.de_vocab)
    make_vocab(hp.target_train, hp.en_vocab)
    print("Done")
