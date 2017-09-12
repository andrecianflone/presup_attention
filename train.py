import tensorflow as tf
from model import Attn
import numpy as np
from sklearn.metrics import accuracy_score
np.random.seed(seed=1)

def make_batches(x, x_len, y, shuffle=True):
  """ Yields the data object with all properties sliced """
  y = one_hot(y)
  data_size = len(x)
  indices = np.arange(0, data_size)
  num_batches = data_size//hp.batch_size+(data_size%hp.batch_size>0)
  if shuffle: np.random.shuffle(indices)
  for batch_num in range(num_batches):
    start_index = batch_num * hp.batch_size
    end_index = min((batch_num + 1) * hp.batch_size, data_size)
    new_indices = indices[start_index:end_index]
    yield (x[new_indices], x_len[new_indices], y[new_indices])

def one_hot(arr):
  """ One-hot encode, where values interpreted as index of non-zero column """
  mat = np.zeros((arr.size, arr.max()+1))
  mat[np.arange(arr.size),arr] = 1
  return mat

def accuracy(sess, teX, teXlen, teY, model):
  """ Return accuracy """
  fetch = [model.batch_size, model.cost, model.y_pred, model.y_true]
  y_pred = np.zeros(teX.shape[0])
  y_true = np.zeros(teX.shape[0])
  start_id = 0
  for batch in make_batches(teX, teXlen, teY, shuffle=False):
    result = call_model(sess, model, batch, fetch, hp.keep_prob, mode=0)
    batch_size                           = result[0]
    cost                                 = result[1]
    y_pred[start_id:start_id+batch_size] = result[2]
    y_true[start_id:start_id+batch_size] = result[3]
    start_id += batch_size

  acc = accuracy_score(y_true, y_pred)
  return acc

def call_model(sess, model, batch, fetch, keep_prob, mode):
  """ Calls models and yields results per batch """
  x     = batch[0]
  x_len = batch[1]
  y     = batch[2]
  feed = {
           model.keep_prob : keep_prob,
           model.mode      : mode, # 1 for train, 0 for testing
           model.inputs    : x,
           model.input_len : x_len,
           model.labels    : y
         }

  result = sess.run(fetch,feed)
  return result

def decoder_mask():
  """ Returns tensor of same shape as decoder output to mask padding """
  ones = np.ones([batch_size,hp.max_seq_len])
  ones[trXlen[:,None] <= np.arange(trXlen.shape[1])] = 0
  np.repeat(d[:, :, np.newaxis], 2, axis=2)

def train_model(params, emb, trX, trXlen, trY, teX, teXlen, teY):
  global hp
  hp = params
  best_acc = 0
  best_epoch = 0
  with tf.Graph().as_default(), tf.Session() as sess:
    tf.set_random_seed(1)
    model = Attn(hp, emb)
    tf.global_variables_initializer().run()
    for epoch in range(hp.max_epochs):
      for batch in make_batches(trX, trXlen, trY, shuffle=True):
        fetch = [model.optimize, model.cost, model.global_step]
        _, cost, step = call_model(sess, model, batch, fetch, hp.keep_prob, mode=1)
        if step%10==0:
          print('ep {} step {} - cost {}'.format(epoch, step, cost))
        if step%20==0:
          acc = accuracy(sess, teX, teXlen, teY, model)
          if acc>best_acc:
            best_acc = acc
            best_epoch = epoch
          print('*********** {}'.format(acc))
    print('Best epoch {}, acc: {}'.format(best_epoch, best_acc))
