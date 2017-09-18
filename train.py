# Author: Andre Cianflone
import tensorflow as tf
from model import PairWiseAttn, AttnAttn, ConvAttn
from utils import Progress, make_batches, calc_num_batches
import numpy as np
from sklearn.metrics import accuracy_score
random_seed=1
# np.random.seed(seed=random_seed)

def train_model(params, emb, trX, trXlen, trY, vaX, vaXlen, vaY, teX,
                                                                  teXlen, teY):
  global hp
  hp = params
  prog = Progress(calc_num_batches(trX, hp.batch_size))
  best_acc = 0
  best_epoch = 0
  te_acc = 0

  # Start tf session
  with tf.Graph().as_default(), tf.Session() as sess:
    tf.set_random_seed(random_seed)
    model = ConvAttn(hp, emb)
    # model = PairWiseAttn(hp, emb)
    tf.global_variables_initializer().run()

    # Begin training and occasional validation
    for epoch in range(hp.max_epochs):
      prog.epoch_start()
      for batch in make_batches(trX, trXlen, trY, hp.batch_size,
                                                    shuffle=True, seed=epoch):
        fetch = [model.optimize, model.cost, model.global_step]
        _, cost, step = call_model(sess, model, batch, fetch, hp.keep_prob, mode=1)
        prog.print_train(cost)
        if step%hp.eval_every==0:
          va_acc = accuracy(sess, vaX, vaXlen, vaY, model)
          if va_acc>best_acc:
            best_acc = va_acc
            best_epoch = epoch
            te_acc = accuracy(sess, teX, teXlen, teY, model)
            prog.test_best_val(te_acc)
          prog.print_eval(va_acc)
    prog.train_end()
    print('Best epoch {}, acc: {}'.format(best_epoch, best_acc))

def accuracy(sess, teX, teXlen, teY, model):
  """ Return accuracy """
  fetch = [model.batch_size, model.cost, model.y_pred, model.y_true]
  y_pred = np.zeros(teX.shape[0])
  y_true = np.zeros(teX.shape[0])
  start_id = 0
  for batch in make_batches(teX, teXlen, teY, hp.batch_size, shuffle=False):
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

