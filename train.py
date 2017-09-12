import tensorflow as tf
from model import Attn
import numpy as np
np.random.seed(seed=1)

def make_batches(x, x_len, y, shuffle=True):
  """ Yields the data object with all properties sliced """
  data_size = len(x)
  indices = np.arange(0, data_size)
  num_batches = data_size//hp.batch_size+(data_size%hp.batch_size>0)
  if shuffle: np.random.shuffle(indices)
  for batch_num in range(num_batches):
    start_index = batch_num * hp.batch_size
    end_index = min((batch_num + 1) * hp.batch_size, data_size)
    new_indices = indices[start_index:end_index]
    yield (x[new_indices], x_len[new_indices], y[new_indices])

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

def train_model(params, emb, trX, trXlen, trY, teX, teXlen, teY):
  global hp
  hp = params
  with tf.Graph().as_default(), tf.Session() as sess:
    tf.set_random_seed(1)
    model = Attn(hp, emb)
    tf.global_variables_initializer().run()
    for batch in make_batches(trX, trXlen, trY, shuffle=True):
      fetch = [model.encoded_outputs]
      result = call_model(sess, model, fetch, keep_prob=1, mode=1)
      pass

