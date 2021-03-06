import numpy as np
import tensorflow as tf
from base_model import BaseModel
from modules import embedding
from modules import feedforward
from modules import positional_encoding
from modules import multihead_attention
from hyperparams import Hyperparams as hp

class TransformerModel(BaseModel):
    def __init__(self, is_training):
        BaseModel.__init__(self, is_training)
    
    def build_model(self):
        # define decoder inputs
        self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1])*2, self.y[:, :-1]), -1) # 2:<S>
        
        # Encoder
        with tf.variable_scope("encoder"):
            ## Embedding
            self.enc = embedding(
                self.x, 
                vocab_size=len(self.de2idx), 
                num_units=hp.emb_dim, 
                scale=True,
                scope="enc_embed")
            sign = tf.sign(tf.reduce_sum(tf.abs(self.enc), axis=-1))
            key_masks = tf.expand_dims(sign, -1)

            ## Positional Encoding
            if hp.sinusoid:
                self.enc += positional_encoding(self.x,
                                  num_units=hp.emb_dim, 
                                  zero_pad=False, 
                                  scale=False,
                                  scope="enc_pe")
            else:
                self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                  vocab_size=hp.maxlen, 
                                  num_units=hp.emb_dim, 
                                  zero_pad=False, 
                                  scale=False,
                                  scope="enc_pe")

            self.enc *= key_masks
             
            ## Dropout
            self.enc = tf.layers.dropout(
                self.enc, 
                rate=hp.dropout_rate, 
                training=tf.convert_to_tensor(self.is_training))
            
            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.enc = multihead_attention(
                        queries=self.enc, 
                        keys=self.enc, 
                        num_units=hp.emb_dim, 
                        num_heads=hp.num_heads, 
                        dropout_rate=hp.dropout_rate,
                        is_training=self.is_training,
                        causality=False)
                    
                    ### Feed Forward
                    self.enc = feedforward(
                        self.enc,
                        num_units=[4*hp.emb_dim,
                                   hp.emb_dim])
        
        # Decoder
        with tf.variable_scope("decoder"):
            ## Embedding
            self.dec = embedding(self.decoder_inputs, 
                                  vocab_size=len(self.en2idx), 
                                  num_units=hp.emb_dim,
                                  scale=True, 
                                  scope="dec_embed")

            key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(self.dec), axis=-1)), -1)

            ## Positional Encoding
            if hp.sinusoid:
                self.dec += positional_encoding(self.decoder_inputs,
                                  num_units=hp.emb_dim, 
                                  zero_pad=False, 
                                  scale=False,
                                  scope="dec_pe")
            else:
                self.dec += embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
                    num_units=hp.emb_dim, 
                    zero_pad=False, 
                    scale=False,
                    scope="dec_pe")
            self.dec *= key_masks
            
            ## Dropout
            self.dec = tf.layers.dropout(
                self.dec, 
                rate=hp.dropout_rate, 
                training=tf.convert_to_tensor(self.is_training))
            
            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention ( self-attention)
                    self.dec = multihead_attention(
                        queries=self.dec, 
                        keys=self.dec, 
                        num_units=hp.emb_dim, 
                        num_heads=hp.num_heads, 
                        dropout_rate=hp.dropout_rate,
                        is_training=self.is_training,
                        causality=True, 
                        scope="self_attention")
                    
                    ## Multihead Attention ( vanilla attention)
                    self.dec = multihead_attention(
                        queries=self.dec, 
                        keys=self.enc, 
                        num_units=hp.emb_dim, 
                        num_heads=hp.num_heads,
                        dropout_rate=hp.dropout_rate,
                        is_training=self.is_training, 
                        causality=False,
                        scope="vanilla_attention")
                    
                    ## Feed Forward
                    self.dec = feedforward(self.dec, num_units=[4*hp.emb_dim, hp.emb_dim])
        
        # Final linear projection
        self.logits = tf.layers.dense(self.dec, len(self.en2idx))
        self.preds = tf.to_int32(tf.argmax(self.logits, dimension=-1))
    
    def get_train_op(self):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        
    @staticmethod
    def get_pred(sess, g, x):
        preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
        for j in range(hp.maxlen):
            _preds = sess.run(g.preds, {g.x: x, g.y: preds})
            preds[:, j] = _preds[:, j]
        return preds
    