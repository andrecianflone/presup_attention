# Author: Andre Cianflone
import tensorflow as tf
from model import PairWiseAttn, AttnAttn, ConvAttn
from utils import Progress, make_batches, calc_num_batches, save_model, load_model
import numpy as np
from pydoc import locate
from sklearn.metrics import accuracy_score
# np.random.seed(seed=random_seed)

def train_model(params, sess, saver, model, result, data):
  trX, trXlen, trY, vaX, vaXlen, vaY, teX, teXlen, teY = data
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
    for batch in make_batches(trX, trXlen, trY, hp.batch_size,
                                                  shuffle=True, seed=epoch):
      fetch = [model.optimize, model.cost, model.global_step]
      _, cost, step = call_model(\
          sess, model, batch, fetch, hp.keep_prob, hp.rnn_in_keep_prob, mode=1)
      prog.print_train(cost)
      if step%hp.eval_every==0:
        va_acc = accuracy(sess, vaX, vaXlen, vaY, model)
        # If best!
        if va_acc>best_acc:
          best_acc = va_acc
          best_epoch = epoch
          te_acc = accuracy(sess, teX, teXlen, teY, model)
          result = {'va_acc':va_acc, 'te_acc':te_acc, 'epoch':epoch}
          save_model(sess, saver, hp, result, step, if_global_best=1)
          prog.test_best_val(te_acc)
        prog.print_eval(va_acc)
    # Early stop check
    if epoch - best_epoch > hp.early_stop: break
  prog.train_end()
  print('Best epoch {}, acc: {}'.format(best_epoch+1, best_acc))

def accuracy(sess, teX, teXlen, teY, model):
  """ Return accuracy """
  fetch = [model.batch_size, model.cost, model.y_pred, model.y_true]
  y_pred = np.zeros(teX.shape[0])
  y_true = np.zeros(teX.shape[0])
  start_id = 0
  for batch in make_batches(teX, teXlen, teY, hp.batch_size, shuffle=False):
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
  x_len = batch[1]
  y     = batch[2]
  feed = {
           model.keep_prob        : keep_prob,
           model.rnn_in_keep_prob : rnn_in_keep_prob,
           model.mode             : mode, # 1 for train, 0 for testing
           model.inputs           : x,
           model.input_len        : x_len,
           model.labels           : y
         }

  result = sess.run(fetch,feed)
  return result

def examine_attn(hp, sess, model, data):
  fetch = [model.col_attn, model.row_attn, model.attn_over_attn, model.y_pred, model.y_true]
  trX, trXlen, trY, vaX, vaXlen, vaY, teX, teXlen, teY = data
  # Grab a sample
  rand = np.random.randint(len(teX))
  # sample = [r_teX, r_teXlen, r_teY] = teX[rand], teXlen[rand], teY[rand]
  sample = [teX[rand], teXlen[rand], teY[rand]]
  col, row, aoa, y_pred, y_true = \
                          call_model(sess, model, batch, sample, 1, 1, mode=0)

  pass
