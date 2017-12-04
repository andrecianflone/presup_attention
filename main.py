# Author: Andre Cianflone

# cmd to test a sample:
# python main.py --load_saved --ckpt_name also_word --mode 0
# debug
# python -m pudb main.py --load_saved --ckpt_name also_word --mode 0
import sys
import os
import tensorflow as tf
import numpy as np
from call_model import train_model, examine_attn
from utils import HParams, load_model, data_info
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
  mode = hp.mode

  # Get data
  emb, word_idx_map, data = load_data(hp.data_dir, hp.pickle)

  inv_vocab =  data_info(emb,word_idx_map,data)

  # Start tf session
  with tf.Graph().as_default(), tf.Session() as sess:
    # Get the model
    model, saver, hp, result = load_model(sess, emb, hp)

    # Check the params
    print(hp)

    # Train the model or examine results
    if mode == 1:
      # Train the model!
      train_model(hp, sess, saver, model, result, data)
    else:
      examine_attn(hp, sess, model, word_idx_map, inv_vocab, data)
    pass


