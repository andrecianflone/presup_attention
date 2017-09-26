# Author: Andre Cianflone
from datetime import datetime
import numpy as np
from numpy.random import RandomState

class Progress():
  """ Pretty print progress for neural net training """
  def __init__(self, batches, progress_bar=True, bar_length=30, track_best=True):
    self.progress_bar = progress_bar # boolean
    self.bar_length = bar_length
    self.t1 = datetime.now()
    self.train_start_time = self.t1
    self.batches = batches
    self.current_batch = 0
    self.epoch = 0
    self.last_eval = '' # save last eval to add after train
    self.last_train = ''
    self.track_best = track_best
    self.best_val = 0
    self.test_val = 0

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

