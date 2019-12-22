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
    
    def rnn_cell(self):
        def single_rnn_cell():
            if hp.rnn_type == 'rnn':
                single_cell = tf.contrib.rnn.BLSTMCell(hp.rnn_size)
            elif hp.rnn_type == 'gru':
                single_cell = tf.contrib.rnn.GRUCell(hp.rnn_size)
            elif hp.rnn_type == 'lstm':
                single_cell = tf.contrib.rnn.LSTMCell(hp.rnn_size)
            cell = tf.contrib.rnn.DropoutWrapper(
                single_cell, output_keep_prob=hp.dropout_rate)
            return cell
        #return single_rnn_cell()
        cell = tf.contrib.rnn.MultiRNNCell(
            [single_rnn_cell() for _ in range(hp.rnn_num_layers)])
        return cell
    
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
                cells = self.rnn_cell()
                encoder_output, _encoder_state = tf.nn.dynamic_rnn(
                    cells, self.enc, sequence_length=self.x_len,
                    dtype=tf.float32)
                self.enc = tf.concat([self.enc, encoder_output], axis=-1)
                self.enc = tf.layers.dense(self.enc, hp.emb_dim, activation="relu")
            
            self.enc *= key_masks
            
            ## Dropout
            self.enc = tf.layers.dropout(
                self.enc, 
                rate=hp.dropout_rate, 
                training=tf.convert_to_tensor(self.is_training))
            
            with tf.variable_scope("enc_pos_emb"):
                pos_emb = tf.get_variable('enc_pos_emb',
                    dtype=tf.float32,
                    shape=[self.enc.shape[1]],
                    initializer=tf.contrib.layers.xavier_initializer())
            
            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.enc = multihead_attention(
                        queries=self.enc, 
                        keys=self.enc,
                        pos_emb=pos_emb,
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
                cells = self.rnn_cell()
                decoder_output, _decoder_state = tf.nn.dynamic_rnn(
                    cells, self.dec,
                    sequence_length=self.y_len,
                    dtype=tf.float32)
                self.dec = tf.concat([self.dec, decoder_output], axis=-1)
                self.dec = tf.layers.dense(self.dec, hp.emb_dim, activation="relu")
                
            self.dec *= key_masks
            
            ## Dropout
            self.dec = tf.layers.dropout(
                self.dec, 
                rate=hp.dropout_rate, 
                training=tf.convert_to_tensor(self.is_training))
            
            with tf.variable_scope("dec_pos_emb"):
                dec_dec_pos_emb = tf.get_variable('dec_de_pos_emb',
                    dtype=tf.float32,
                    shape=[self.dec.shape[1]],
                    initializer=tf.contrib.layers.xavier_initializer())
                dec_enc_pos_emb = tf.get_variable('dec_enc_pos_emb',
                    dtype=tf.float32,
                    shape=[self.enc.shape[1]],
                    initializer=tf.contrib.layers.xavier_initializer())
            
            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention ( self-attention)
                    self.dec = multihead_attention(
                        queries=self.dec, 
                        keys=self.dec, 
                        pos_emb=dec_dec_pos_emb,
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
                        pos_emb=dec_enc_pos_emb,
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
    
