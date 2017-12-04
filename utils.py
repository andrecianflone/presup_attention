# Author: Andre Cianflone
from datetime import datetime
from pprint import pformat, pprint
import os, argparse, pickle, json, tarfile, copy, sys
from pydoc import locate
import tensorflow as tf
import numpy as np
from numpy.random import RandomState

class Progress():
  """ Pretty print progress for neural net training """
  def __init__(self, batches, best_val=0, test_val=0, epoch=0,
                            progress_bar=True, bar_length=30, track_best=True):
    self.progress_bar = progress_bar # boolean
    self.bar_length = bar_length
    self.t1 = datetime.now()
    self.train_start_time = self.t1
    self.batches = batches
    self.current_batch = 0
    self.last_eval = '' # save last eval to add after train
    self.last_train = ''
    self.track_best = track_best
    self.best_val = best_val
    self.test_val = test_val
    self.epoch = epoch

  def epoch_start(self):
    print()
    self.t1 = datetime.now()
    self.epoch += 1
    self.current_batch = 0 # reset batch

  def train_end(self):
    print()

  def print_train(self, loss):
    t2 = datetime.now()
    epoch_time = (t2 - self.t1).total_seconds()
    total_time = (t2 - self.train_start_time).total_seconds()/60
    self.last_train='{:2.0f}: sec: {:>5.0f} | total min: {:>5.1f} | train loss: {:>3.4f} '.format(
        self.epoch, epoch_time, total_time, loss)
    print(self.last_train, end='')
    self.print_bar()
    print(self.last_eval, end='\r')

  def print_cust(self, msg):
    """ Print anything, append previous """
    print(msg, end='')

  def test_best_val(self, te_acc):
    """ Test set result at the best validation checkpoint """
    self.test_val = te_acc

  def print_eval(self, value):
    # Print last training info
    print(self.last_train, end='')
    self.last_eval = '| last val: {:>3.4f} '.format(value)

    # If tracking eval, update best
    extra = ''
    if self.track_best == True:
      if value > self.best_val:
        self.best_val = value
      self.last_eval += '| best val: {:>3.4f} | test on best model: {:>3.4f}'.format(self.best_val, self.test_val)
    print(self.last_eval, end='\r')

  def print_bar(self):
    self.current_batch += 1
    bars_full = int(self.current_batch/self.batches*self.bar_length)
    bars_empty = self.bar_length - bars_full
    progress ="| [{}{}] ".format(u"\u2586"*bars_full, '-'*bars_empty)
    self.last_train += progress
    print(progress, end='')

def seq_length(sequence):
  """ Compute length on the fly in the graph, sequence is [batch, time, dim]"""
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length

def make_batches(x, x_len, y, batch_size, shuffle=True, seed=0):
  """ Yields the data object with all properties sliced """
  y = one_hot(y)
  data_size = len(x)
  indices = np.arange(0, data_size)
  num_batches = data_size//batch_size+(data_size%batch_size>0)
  if shuffle:
    rnd = RandomState(seed) # repeatable shuffle
    rnd.shuffle(indices)
  for batch_num in range(num_batches):
    start_index = batch_num * batch_size
    end_index = min((batch_num + 1) * batch_size, data_size)
    new_indices = indices[start_index:end_index]
    yield (x[new_indices], x_len[new_indices], y[new_indices])

def calc_num_batches(x, batch_size):
  """ Return number of batches for this set """
  data_size = len(x)
  num_batches = data_size//batch_size+(data_size%batch_size>0)
  return num_batches

def one_hot(arr):
  """ One-hot encode, where values interpreted as index of non-zero column """
  mat = np.zeros((arr.size, arr.max()+1))
  mat[np.arange(arr.size),arr] = 1
  return mat

def decoder_mask():
  """ Returns tensor of same shape as decoder output to mask padding """
  ones = np.ones([batch_size,hp.max_seq_len])
  ones[trXlen[:,None] <= np.arange(trXlen.shape[1])] = 0
  np.repeat(d[:, :, np.newaxis], 2, axis=2)


class HParams():
  def __init__(self):
    parser = argparse.ArgumentParser(description='Presupposition attention')

    # General flags
    add = parser.add_argument
    add('--data_dir', type=str, default="../presup_giga_also/")
    add('--pickle', type=str, default="processed.pkl")
    # add('--pickle', type=str, default="processed_singleUnk.pkl")
    add('--model', type=str, default="AttnAttn")
    add('--load_saved', action='store_true', default=False)
    add('--ckpt_dir', type=str, default='ckpt')
    add('--ckpt_name', type=str, default='ckpt')
    add('--mode', type=int, default=1, help='train: 1, test:0')

    # Hyperparams
    add('--emb_trainable', action='store_true', default=False)
    add('--batch_size', type=int, default=64)
    add('--max_seq_len', type=int, default=60)
    add('--max_epochs', type=int, default= 50)
    add('--early_stop', type=int, default= 10)
    add('--rnn_in_keep_prob', type=float, default=1.0)
    add('--word_gate', action='store_true', default=False)
    # Variational recurrent: if true, same rnn drop mask at each step
    add('--variational_recurrent', action='store_true', default = False)
    add('--keep_prob', type=float, default=0.5)
    add('--eval_every', type=int, default=300)
    add('--num_classes', type=int, default=2)
    add('--l_rate', type=float, default= 0.001)
    add('--cell_units', type=int, default=128)
    add('--cell_type', type=str, default='LSTMCell')
    add('--optimizer', type=str, default='AdamOptimizer')

    # Hyper params for dense layers
    add('--h_layers', type=int, default=0)
    add('--fc_units', type=int, default=64)

    # Hyper params for convnet
    add('--batch_norm', action='store_true', default=False)
    add('--filt_height', type=int, default=3)
    add('--filt_width', type=int, default=3)
    # For conv_stride: since input is "NHWC", no batch/channel stride
    add('--conv_strides', nargs=4, default=[1,1,1,1])
    add('--padding', type=str, default="VALID")
    add('--out_channels', type=int, default=32)

    args = parser.parse_args()
    self._init_attributes(args)

  def _init_attributes(self, args):
    for k,v in vars(args).items():
      setattr(self, k, v)

  def update(self, k,v):
    setattr(self, k, v)

  def __str__(self):
    return pformat(vars(self),indent=0)

def save_model(sess, saver, hp, result, step, if_global_best=1):
  """
  Args:
    saver: tf saver object
    hp: HParams object used to created the net
    va_acc: validation accuracy
    directory: where to save
    name: name
    if_global_best: if true, overwrites best model in directory if better
  """
  directory = hp.ckpt_dir
  name = hp.ckpt_name
  if not os.path.exists(directory): os.makedirs(directory)
  path = directory + "/" + name
  hp_path = path+"_hp.pkl"
  result_pkl = path+"_result.pkl"
  info = copy.deepcopy(result)
  info['passed args'] = sys.argv

  # Save files temporarily to disk
  pickle.dump(hp, open(hp_path, "wb"))
  pickle.dump(result, open(result_pkl, "wb"))
  with open(path+"_info.json", "w") as f: json.dump(info, f)
  model_path = saver.save(sess, path+"_model.ckpt", global_step=step)

  # Tar the data
  tar_name = path+".tar"
  os.remove(tar_name) if os.path.exists(tar_name) else None
  tar = tarfile.open(tar_name, "w")
  for f in os.listdir(directory):
    if '.tar' in f:continue # don't delete tar!
    if name in f:
      tar.add(directory+"/"+f)
      os.remove(directory+"/"+f)
  ckpt_file = directory+"/"+'checkpoint'
  tar.add(ckpt_file) # add mysterious file
  os.remove(ckpt_file) if os.path.exists(ckpt_file) else None
  tar.close()

def data_info(emb,word_idx_map,data):
  # Create inverse vocab, mapping integer to word
  inv_vocab = {}
  for k,v in word_idx_map.items():
    inv_vocab[v] = k

  unk_idx = 112368
  unk_emb = emb[112368]
  unk_emb2 = emb[143661]

  return inv_vocab

def load_model(sess, emb, hp):
  """ Returns new model or presaved model depending on hyperparams"""
  dirt, name, load_saved = hp.ckpt_dir, hp.ckpt_name, hp.load_saved

  # If new, returns new model
  if load_saved == False:
    model = locate("model." + hp.model)
    model = model(hp, emb)
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    print("New model initialized")
    return model, saver, hp, None

  # Find saved weights
  tar_path = dirt + "/" + name + ".tar"
  tar = tarfile.open(tar_path)
  # Get paths
  for member in tar.getmembers():
    if 'hp.pkl' in member.name: hparams = member
    if 'result.pkl' in member.name: result = member
    if 'meta' in member.name: meta = member
    if 'data' in member.name: variables = member
  tar.extractall()

  # Get params
  hp = pickle.load(open(hparams.name, "rb"))
  hp.update('ckpt_dir', dirt)
  hp.update('name', name)

  # Get previous results
  result = pickle.load(open(result.name, "rb"))
  # result = None

  # Restore model
  model = locate("model." + hp.model)
  model = model(hp, emb)
  tf.global_variables_initializer().run()

  # Restore variables
  model_path = variables.name.split('.data')[0]
  saver = tf.train.Saver()
  saver.restore(sess, model_path)
  print("*"*79)
  print("Successfully restored previous model")
  print("*"*79)

  # Remove temp files
  for member in tar.getmembers():
    os.remove(member.name) if os.path.exists(member.name) else None
  tar.close()

  return model, saver, hp, result

