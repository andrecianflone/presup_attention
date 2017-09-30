# Author: Andre Cianflone
import sys
import os
import tensorflow as tf
import numpy as np
from train import train_model
from utils import HParams, load_model
# Ugly hack: add parent as package to allow relative imports
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from CNN_sentence.CNN_sentence import load_data
# Control repeatability
random_seed=1
tf.set_random_seed(random_seed)

# TODO: check the embedding and integerization. Why does 'the'
# refer to index 5273? Should be one of the first
# TODO: Implement Batch norm mLSTM?

if __name__=="__main__":
  # Get hyperparams from argparse and defaults
  hp = HParams()

  # Get data
  emb, word_idx_map, data = load_data(hp.data_dir)

  # Start tf session
  with tf.Graph().as_default(), tf.Session() as sess:
    # Get the model
    model, saver, hp, result = load_model(sess, emb, hp)

    print(hp)

    # Train the model!
    train_model(hp, sess, saver, model, result, data)
