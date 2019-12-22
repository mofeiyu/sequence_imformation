# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function

import codecs
import tensorflow as tf
from hyperparams import Hyperparams as hp
from nltk.translate.bleu_score import corpus_bleu
from data_load import load_test_data, load_en_vocab

def evaluate(g, maxstep=0):
    # Load data
    X, Sources, Targets = load_test_data()
    _en2idx, idx2en = load_en_vocab()
    
    # Start session         
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
            
            ## Inference
            with codecs.open('result.txt', "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                for i in range(len(X) // hp.batch_size):
                    if maxstep > 0 and i >= maxstep:
                        break
                    
                    ### Get mini-batches
                    x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                    sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                    targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]
                     
                    ### Autoregressive inference
                    preds = g.get_pred(sess, g, x)
                    
                    ### Write to file
                    for source, target, pred in zip(sources, targets, preds): # sentence-wise
                        got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                        fout.write("- source: " + source +"\n")
                        fout.write("- expected: " + target + "\n")
                        fout.write("- got: " + got + "\n\n")
                        fout.flush()
                        
                        # bleu score
                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)
              
                ## Calculate bleu score
                if len(list_of_refs) > 100:
                    score = corpus_bleu(list_of_refs, hypotheses)
                    s = "Bleu Score = " + str(100*score)
                    print (s)
                    fout.write(s)
                    