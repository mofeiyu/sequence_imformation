# -*- coding: utf-8 -*-
#/usr/bin/python3

import tensorflow as tf

from model import Transformer
from tqdm import tqdm
from data_load import get_batch, load_data_eval
from utils import save_hparams, save_variable_specs, get_hypotheses, calc_bleu
import os
from hparams import Hparams
import math
import logging

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

logging.info("# Prepare train/eval batches")
train_batches, num_train_batches, num_train_samples = get_batch(hp.train1, hp.train2,
                                             hp.maxlen1, hp.maxlen2,
                                             hp.vocab, hp.batch_size,
                                             shuffle=True)
eval_batches, num_eval_batches, num_eval_samples = get_batch(hp.eval1, hp.eval2,
                                             hp.maxlen1, hp.maxlen2,
                                             hp.vocab, hp.batch_size,
                                             shuffle=False)
load_data_eval(hp.eval1, hp.eval2, hp.eval33, hp.maxlen1, hp.maxlen2)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys = iter.get_next()

train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)

m = Transformer(hp)

logging.info("# Load model")
y_hat, eval_summaries = m.eval(xs, ys)
# y_hat = m.infer(xs, ys)


logging.info("# Session")
saver = tf.train.Saver()
with tf.Session() as sess:
    #if Hparams.prefix.startswith('/'):
    #    ckpt = tf.train.latest_checkpoint('/logdir')
    #else:
    ckpt = tf.train.latest_checkpoint('logdir')
        
    saver.restore(sess, ckpt)
    
    sess.run(train_init_op)
    
    logging.info("# test evaluation")
    _, _eval_summaries = sess.run([eval_init_op, eval_summaries])
    
    logging.info("# get hypotheses")
    hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat, m.idx2token)
    
    logging.info("# write results")
    model_output = "eval_file"
    if not os.path.exists(hp.evaldir): os.makedirs(hp.evaldir)
    translation = os.path.join(hp.evaldir, model_output)
    with open(translation, 'w') as fout:
        fout.write("\n".join(hypotheses))

    logging.info("# calc bleu score and append it to translation")
    calc_bleu(hp.eval3, translation)

    logging.info("# save models")
    ckpt_name = os.path.join(hp.logdir, model_output)

logging.info("Done")
