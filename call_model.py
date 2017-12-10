# Author: Andre Cianflone
import tensorflow as tf
from model import PairWiseAttn, AttnAttn, ConvAttn
from utils import Progress, make_batches, calc_num_batches, save_model, load_model, one_hot
import numpy as np
from pydoc import locate
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# np.random.seed(seed=random_seed)

def train_model(params, sess, saver, model, result, data):
  trX, trXTags, trXlen, trY, vaX, vaXTags, vaXlen, vaY, teX, teXTags, teXlen, teY = data
  global hp
  hp = params
  if result is not None:
    best_acc = result['va_acc']
    te_acc = result['te_acc']
    epoch = result['epoch']
  else:
    best_acc = 0
    te_acc = 0
    epoch = 0
  prog = Progress(calc_num_batches(trX, hp.batch_size), best_acc, te_acc)
  best_epoch = 0

  # Begin training and occasional validation
  for epoch in range(epoch, epoch+hp.max_epochs):
    prog.epoch_start()
    for batch in make_batches(trX, trXTags, trXlen, trY, hp.batch_size,
                                                  shuffle=True, seed=epoch):
      fetch = [model.optimize, model.cost, model.global_step]
      _, cost, step = call_model(\
          sess, model, batch, fetch, hp.keep_prob, hp.rnn_in_keep_prob, mode=1)
      prog.print_train(cost)
      if step%hp.eval_every==0:
        va_acc = accuracy(sess, vaX, vaXTags, vaXlen, vaY, model)
        # If best!
        if va_acc>best_acc:
          best_acc = va_acc
          best_epoch = epoch
          te_acc = accuracy(sess, teX, teXTags, teXlen, teY, model)
          result = {'va_acc':va_acc, 'te_acc':te_acc, 'epoch':epoch}
          save_model(sess, saver, hp, result, step, if_global_best=1)
          prog.test_best_val(te_acc)
        prog.print_eval(va_acc)
    # Early stop check
    if epoch - best_epoch > hp.early_stop: break
  prog.train_end()
  print('Best epoch {}, acc: {}'.format(best_epoch+1, best_acc))

def accuracy(sess, teX, teXTags, teXlen, teY, model):
  """ Return accuracy """
  fetch = [model.batch_size, model.cost, model.y_pred, model.y_true]
  y_pred = np.zeros(teX.shape[0])
  y_true = np.zeros(teX.shape[0])
  start_id = 0
  for batch in make_batches(teX, teXTags, teXlen, teY, hp.batch_size, shuffle=False):
    result = call_model(sess, model, batch, fetch, 1, 1, mode=0)
    batch_size                           = result[0]
    cost                                 = result[1]
    y_pred[start_id:start_id+batch_size] = result[2]
    y_true[start_id:start_id+batch_size] = result[3]
    start_id += batch_size

  acc = accuracy_score(y_true, y_pred)
  return acc

def call_model(sess, model, batch, fetch, keep_prob, rnn_in_keep_prob, mode):
  """ Calls models and yields results per batch """
  x     = batch[0]
  x_tags= batch[1]
  x_len = batch[2]
  y     = batch[3]
  feed = {
           model.keep_prob        : keep_prob,
           model.rnn_in_keep_prob : rnn_in_keep_prob,
           model.mode             : mode, # 1 for train, 0 for testing
           model.inputs           : x,
           model.postags          : x_tags,
           model.input_len        : x_len,
           model.labels           : y
         }

  result = sess.run(fetch,feed)
  return result

def sample_to_sent(x, inv_vocab):
  """ Swap integers in `x` for words, retun list of words"""
  inv_vocab[0] = '<pad>'
  sample = [inv_vocab[w] for w in x]
  return sample

def chart_sent(sent, attn, y_pred, y_true):
  """ Bar charts the attention with words in `sent` as x tick labels """
  x = np.arange(len(sent))
  plt.bar(x, attn, width=1)
  plt.xticks(x, sent, rotation='vertical')
  label = 'pred: {}, true: {}'.format(y_pred, y_true)
  plt.xlabel(label, fontsize=14)
  plt.margins(0.2)
  plt.subplots_adjust(bottom=0.15)
  plt.show()

def examine_attn(hp, sess, model, vocab, inv_vocab, data):
  fetch = [model.col_attn, model.row_attn, model.attn_over_attn, model.y_pred, model.y_true]
  trX, trXTags, trXlen, trY, vaX, vaXTags, vaXlen, vaY, teX, teXTags, teXlen, teY = data
  # Grab a random sample
  rand = np.random.randint(len(teX))
  y = one_hot(teY)
  sample = (teX[rand:rand+2], teXTags[rand:rand+2], teXlen[rand:rand+2], y[rand:rand+2])
  # sample = (np.expand_dims(teX[rand], axis=0),
            # np.expand_dims(teXlen[rand], axis=0),
            # np.expand_dims(one_hot(teY[rand]),axis=0)
            # )
  col, row, aoa, y_pred, y_true = \
                            call_model(sess, model, sample, fetch, 1, 1, mode=0)
  # Parse sample to text
  sent = sample_to_sent(sample[0][0], inv_vocab)
  chart_sent(sent, aoa[0], y_pred[0], y_true[0])
  print(sent)
  pass
