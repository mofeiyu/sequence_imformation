import os
import shutil
from transformer_model import TransformerModel
from seq2seq_model import Seq2SeqModel
from train import train
from evaluate import evaluate
from hyperparams import Hyperparams as hp

def run(model):
    maxstep = 0
    
    if 1:
        if maxstep > 0 and os.path.exists(hp.logdir):
            shutil.rmtree(hp.logdir)
        
        print ('start train')
        g = model(is_training=True)
        g.init()
        train(g, maxstep)
    
    if 1:
        print ('start evaluate')
        g = model(is_training=False)
        g.init()
        evaluate(g, maxstep)
    print ('main end')

if __name__ == '__main__':
    if 1:
        run(TransformerModel)
    else:
        run(Seq2SeqModel)

