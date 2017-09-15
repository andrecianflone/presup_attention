# Author: Andre Cianflone
from datetime import datetime

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
    self.last_train='{:2.0f}: sec: {:>5.1f} | total min: {:>5.1f} | train loss: {:>3.4f} '.format(
        self.epoch, epoch_time, total_time, loss)
    print(self.last_train, end='')
    self.print_bar()
    print(self.last_eval, end='\r')

  def print_cust(self, msg):
    """ Print anything, append previous """
    print(msg, end='')

  def print_eval(self, msg, value):
    # Print last training info
    print(self.last_train, end='')
    self.last_eval = '| {}: {:>3.4f} '.format(msg, value)

    # If tracking eval, update best
    extra = ''
    if self.track_best == True:
      if value > self.best_val:
        self.best_val = value
      self.last_eval += '| {}: {:>3.4f} '.format('best val', self.best_val)
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

