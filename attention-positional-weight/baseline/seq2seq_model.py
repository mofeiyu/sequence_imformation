import numpy as np
import tensorflow as tf
from base_model import BaseModel
from hyperparams import Hyperparams as hp

class Seq2SeqModel(BaseModel):
    def __init__(self, is_training):
        BaseModel.__init__(self, is_training)
        self.beam_width = hp.rnn_beam_width
        self.rnn_type = 'lstm'
    
    def rnn_cell(self):
        def single_rnn_cell():
            if self.rnn_type == 'rnn':
                single_cell = tf.contrib.rnn.BLSTMCell(hp.rnn_size)
            elif self.rnn_type == 'gru':
                single_cell = tf.contrib.rnn.GRUCell(hp.rnn_size)
            elif self.rnn_type == 'lstm':
                single_cell = tf.contrib.rnn.LSTMCell(hp.rnn_size)
            cell = tf.contrib.rnn.DropoutWrapper(
                single_cell, output_keep_prob=hp.dropout_rate)
            return cell
        #return single_rnn_cell()
        cell = tf.contrib.rnn.MultiRNNCell(
            [single_rnn_cell() for _ in range(hp.rnn_num_layers)])
        return cell
    
    def build_model(self):
        # Encoder
        with tf.variable_scope("encoder"):
            cells = self.rnn_cell()
            # en_embedding: [dict_size, dim]
            en_embedding = tf.get_variable('en_emb', [len(self.de2idx), hp.emb_dim])
            # self.x: [batch_size, sents_len, dim]
            self.encoder_emb_inp = tf.nn.embedding_lookup(en_embedding, self.x)
            # self.encoder_output: [batch_size, sents_len, cell_size]
            # self.encoder_state: [batch_size, cell_size]
            self.encoder_output, self.encoder_state = tf.nn.dynamic_rnn(
                cells, self.encoder_emb_inp, sequence_length=self.x_len, dtype=tf.float32)
        
        # Decoder
        with tf.variable_scope("decoder") as decoder_scope:
            projection_layer = tf.layers.Dense(len(self.en2idx), use_bias=False)
            decoder_cell = self.rnn_cell()
            # de_embedding: [dict_size, dim]
            de_embedding = tf.get_variable('dec_emb', [len(self.en2idx), hp.emb_dim])
            
            if self.is_training:
                tiled_encoder_output = self.encoder_output
                tiled_encoder_final_state = self.encoder_state
                tiled_seq_len = self.x_len
                _batch_size = self.batch_size
            else:
                tiled_encoder_output = tf.contrib.seq2seq.tile_batch(
                    self.encoder_output, multiplier=self.beam_width)
                tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
                    self.encoder_state, multiplier=self.beam_width)
                tiled_seq_len = tf.contrib.seq2seq.tile_batch(
                    self.x_len, multiplier=self.beam_width)
                _batch_size = self.batch_size * self.beam_width
            
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                hp.rnn_size, memory=tiled_encoder_output,
                memory_sequence_length=tiled_seq_len, normalize=True)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, attention_mechanism,
                attention_layer_size=hp.rnn_size)
            initial_state = decoder_cell.zero_state(
                dtype=tf.float32, batch_size=_batch_size)
            initial_state = initial_state.clone(
                cell_state=tiled_encoder_final_state)
            start_token = self.en2idx.get('<S>')
            
            if self.is_training:
                start_idx = tf.ones_like(self.y[:, :1]) * start_token
                decoder_inputs = tf.concat((start_idx, self.y[:, :-1]), -1)
                self.decoder_emb_inp = tf.nn.embedding_lookup(
                    de_embedding, decoder_inputs)
                helper = tf.contrib.seq2seq.TrainingHelper(
                    self.decoder_emb_inp, self.y_len)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, helper, initial_state,
                    output_layer=projection_layer)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, impute_finished=True,
                    maximum_iterations=hp.maxlen-1, scope=decoder_scope)
                self.logits = outputs.rnn_output
                pad_shape = hp.maxlen - tf.shape(self.logits)[1]
                shape = [self.batch_size, pad_shape, len(self.en2idx)]
                pad_val = tf.zeros(shape, dtype= tf.float32)
                self.logits = tf.concat([self.logits, pad_val], axis=1)
                self.preds = tf.to_int32(tf.argmax(self.logits, dimension=-1))
            else:
                start_tokens = tf.fill([self.batch_size], start_token)
                end_token = self.en2idx['</S>']
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=de_embedding,
                    start_tokens=start_tokens,
                    end_token=end_token,
                    initial_state=initial_state,
                    beam_width=self.beam_width,
                    output_layer=projection_layer
                )
                
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, maximum_iterations=hp.maxlen, scope=decoder_scope)
                self.preds = outputs.predicted_ids[:,:,0]

    def get_train_op(self):
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
        self.train_op = optimizer.apply_gradients(
            zip(clipped_gradients, params),
            global_step=self.global_step)
    
    @staticmethod
    def get_pred(sess, g, x):
        preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
        return sess.run(g.preds, {g.x: x, g.y: preds})
