import tensorflow as tf
from data_load import load_de_vocab
from data_load import load_en_vocab
from data_load import get_batch_data
from modules import label_smoothing
from hyperparams import Hyperparams as hp

class BaseModel:
    def __init__(self, is_training):
        self.de2idx, _idx2de = load_de_vocab()
        self.en2idx, _idx2en = load_en_vocab()
        self.is_training = is_training
        self.graph = tf.Graph()
        with self.graph.as_default():
            if self.is_training:
                self.x, self.y, self.num_batch = get_batch_data() # (N, T)
            else: # inference
                self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            self.x_len = tf.reduce_sum(self.x, axis=-1)
            self.y_len = tf.reduce_sum(self.y, axis=-1)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.batch_size = tf.shape(self.x)[0]
            
    def init(self):
        with self.graph.as_default():
            self.build_model()
            
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            norm_len = tf.reduce_sum(self.istarget)
            equal_val = tf.to_float(tf.equal(self.preds, self.y))
            self.acc = tf.reduce_sum(equal_val*self.istarget)/ norm_len
            tf.summary.scalar('acc', self.acc)
            tf.summary.scalar('target_norm_len', norm_len)
            
            if self.is_training:
                # Loss
                self.y_smoothed = label_smoothing(
                    tf.one_hot(self.y, depth=len(self.en2idx)))
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.logits, labels=self.y_smoothed)
                a = tf.reduce_sum(self.loss * self.istarget)
                self.loss = a / (tf.reduce_sum(self.istarget))
                
                # Training Scheme
                self.get_train_op()
                
                # Summary
                tf.summary.scalar('mean_loss', self.loss)
                self.merged = tf.summary.merge_all()
