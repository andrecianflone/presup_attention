# Author: Andre Cianflone
import sys
import os
import numpy as np
from train import train_model
# Hack: add parent as package to allow relative imports
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from CNN_sentence.CNN_sentence import load_data

class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

  def update(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

# TODO: check the embedding and integerization. Why does 'the'
# refer to index 5273? Should be one of the first
# TODO: add Conv model batch norm before relu
# TODO: Implement Batch norm mLSTM?

if __name__=="__main__":
  # Get data
  path="../presup_giga_also/"
  # path="../presup_wsj/"
  emb, word_idx_map, trX, trXlen, trY, vaX, vaXlen, vaY, teX, teXlen, teY =\
                                                                load_data(path)

  # Data info
  print('size of sets:')
  print('training: positive: {} negative: {}'.format(\
      np.sum(trY), len(trY)-np.sum(trY)))
  print('validation: positive: {} negative: {}'.format(\
      np.sum(vaY), len(vaY)-np.sum(vaY)))
  print('testing: positive: {} negative: {}'.format(\
      np.sum(teY), len(teY)-np.sum(teY)))

  # General hyper params
  hp = HParams(
    emb_trainable         = False,
    batch_size            = 64,
    max_seq_len           = 60,
    max_epochs            = 20,
    early_stop            = 10,
    rnn_in_keep_prob      = 1.0,
    variational_recurrent = False, # if true, same rnn drop mask at each step
    keep_prob             = 0.5,
    eval_every            = 300,
    num_classes           = 2,
    l_rate                = 0.001,
    cell_units            = 512,
    cell_type             = 'LSTMCell',
    optimizer             = 'AdamOptimizer'
  )

  # Hyper params for dense layers
  hp.update(
    h_layers     = 0,
    dense_units = 64
  )
  # Hyper params for convnet
  hp.update(
    batch_norm   = False,
    filt_height  = 3,
    filt_width   = 3,
    h_units = hp.dense_units,
    conv_strides = [1,2,2,1], #since input is "NHWC", no batch/channel stride
    padding      = "VALID",
    out_channels = 32
  )

  train_model(hp, emb, trX, trXlen, trY, vaX, vaXlen, vaY, teX, teXlen, teY)
