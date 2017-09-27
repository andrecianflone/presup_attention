# Author: Andre Cianflone
import sys
import os
import numpy as np
from train import train_model
from utils import HParams
from pydoc import locate
# Ugly hack: add parent as package to allow relative imports
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from CNN_sentence.CNN_sentence import load_data


# TODO: check the embedding and integerization. Why does 'the'
# refer to index 5273? Should be one of the first
# TODO: Implement Batch norm mLSTM?

if __name__=="__main__":

  # Get hyperparams from argparse and defaults
  hp = HParams()
  # Get data
  emb, word_idx_map, trX, trXlen, trY, vaX, vaXlen, vaY, teX, teXlen, teY =\
                                                        load_data(hp.data_dir)

  # Data info
  print('size of sets:')
  print('training: positive: {} negative: {}'.format(\
      np.sum(trY), len(trY)-np.sum(trY)))
  print('validation: positive: {} negative: {}'.format(\
      np.sum(vaY), len(vaY)-np.sum(vaY)))
  print('testing: positive: {} negative: {}'.format(\
      np.sum(teY), len(teY)-np.sum(teY)))

  model = locate("model." + hp.model)
  model = model(hp, emb)

  # Train the model!
  train_model(hp, model, trX, trXlen, trY, vaX, vaXlen, vaY, teX, teXlen, teY)
