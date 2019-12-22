import os
import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    prefix = '/data/wmt14/'
    if not os.path.exists(prefix):
        prefix = '../../wmt14/small_data/'
    
    # prepro
    parser.add_argument('--vocab_size', default=37000, type=int)

    # train
    ## files
    parser.add_argument('--train1', default=prefix + '/segmented/train.en.bpe',
                             help="german training segmented data")
    parser.add_argument('--train2', default=prefix + '/segmented/train.de.bpe',
                             help="english training segmented data")
    parser.add_argument('--eval1', default=prefix + '/segmented/eval.en.bpe',
                             help="german evaluation segmented data")
    parser.add_argument('--eval2', default=prefix + '/segmented/eval.de.bpe',
                             help="english evaluation segmented data")
    parser.add_argument('--eval3', default='eval3',
                             help="english evaluation unsegmented data")
    parser.add_argument('--eval33', default=prefix + '/prepro/eval.de',
                             help="english evaluation unsegmented data")
    
    ## vocabulary
    parser.add_argument('--vocab', default=prefix + '/segmented/bpe.vocab',
                        help="vocabulary file path")

    # training scheme
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=64, type=int)

    parser.add_argument('--lr', default=1, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=16000, type=int)
    parser.add_argument('--logdir', default="logdir", help="log directory")
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--evaldir', default="eval", help="evaluation dir")

    # model
    parser.add_argument('--d_model', default=512, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen1', default=64, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--maxlen2', default=64, type=int,
                        help="maximum length of a target sequence")
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")

    # test
    parser.add_argument('--test1', default=prefix + '/segmented/test.en.bpe',
                        help="german test segmented data")
    parser.add_argument('--test2', default=prefix + '/prepro/test.de',
                        help="english test data")
    parser.add_argument('--ckpt', help="checkpoint file path")
    parser.add_argument('--test_batch_size', default=64, type=int)
    parser.add_argument('--testdir', default="test/1", help="test result dir")
