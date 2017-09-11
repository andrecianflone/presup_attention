import sys
import os
import numpy as np
from train import train_model
# Hack, add parent as package to allow relative imports
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
if __name__=="__main__":
  # Get data
  path="../presup_wsj/"
  emb, word_to_id, trX, trY, teX, teY = load_data(path)

  # Data info
  print('size of sets:')
  print('training: positive: {} negative: {}'.format(\
      np.sum(trY), len(trY)-np.sum(trY)))
  print('testing: positive: {} negative: {}'.format(\
      np.sum(teY), len(teY)-np.sum(teY)))

  # Hyper params
  hp = HParams(
    emb_trainable  = True,
    batch_size     = 32,
    max_seq_len    = 68,
    num_classes = 2,
    cell_units     = 32,
    cell_type      = 'LSTMCell',
    optimizer      = 'AdamOptimizer'
  )

  train_model(hp, emb, trX, trY, teX, teY)
