# -*- coding: utf-8 -*-
#/usr/bin/python2
from __future__ import print_function

from tqdm import tqdm
import tensorflow as tf
from hyperparams import Hyperparams as hp

def train(g, maxstep=0):
    sv = tf.train.Supervisor(graph=g.graph, 
                             logdir=hp.logdir,
                             save_model_secs=0)
    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs+1):
            if sv.should_stop(): break
            
            with tqdm(total=g.num_batch) as pbar:
                for step in xrange(g.num_batch):
                    if step % 20 == 0:
                        pbar.update(20)
                    sess.run(g.train_op)
                    if maxstep > 0 and step >= maxstep:
                        break
            gs = sess.run(g.global_step)
            print ('epoch {}, global_step {}'.format(epoch, gs))
            sv.saver.save(sess, hp.logdir + '/model_%d' % (epoch))

    print("Done")
