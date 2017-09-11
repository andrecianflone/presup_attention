
from pydoc import locate
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as glorot

class Attn():
  """ BiRNN + Pair-wise Attn + Conv """
  def __init__(self,params, embedding):
    """
    Args:
      hparams: hyper param instance
    """
    global hp
    hp = params

    # helper variable to keep track of steps
    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    self.floatX = tf.float32
    self.intX = tf.int32
    self.bi_encoder_hidden = hp.cell_units * 2

    ############################
    # Inputs
    ############################
    self.keep_prob = tf.placeholder(self.floatX)
    self.mode = tf.placeholder(tf.bool, name="mode") # 1 stands for training
    self.vocab_size = embedding.shape[0]
    # Embedding tensor is of shape [vocab_size x embedding_size]
    self.embedding_tensor = self.embedding_setup(embedding, hp.emb_trainable)

    # RNN inputs
    self.inputs = tf.placeholder(self.intX, shape=[None, hp.max_seq_len])
    self.embedded = self.embedded(self.inputs, self.embedding_tensor)
    # self.embedded = tf.layers.batch_normalization(embedded, training=self.mode)
    self.input_len = tf.placeholder(self.intX, shape=[None,])

    # Targets
    self.labels = tf.placeholder(self.intX, shape=[None, hp.num_classes])

    self.batch_size = tf.shape(self.inputs)[0]

    ############################
    # Build model
    ############################
    # Forward/backward cells
    cell_fw, cell_bw = self.build_cell()

    # Get encoded inputs
    self.encoded_outputs, self.encoded_state = self.encoder_bi(cell_fw,
                                cell_bw, self.embedded, self.input_len)

    # Pair-wise score
    p_w = self.pair_wise_matching(self.encoded_outputs)

    # Attn matrices
    col_attn, row_attn = self.attn_matrices(p_w, self.input_len,
                                                          self.batch_size, dim)

    # Simple output layer
    # x = dense(x, in_dim, out_dim, act=tf.nn.relu, scope=layer_name)
    # x = tf.nn.dropout(x, self.keep_prob)
    logits = dense(x, out_dim, num_classes, act=None, scope="class_log")

    ############################
    # Loss/Optimize
    ############################
    # Build loss
    self.loss = self.classification_loss(self.labels, self.logits)
    self.cost = tf.reduce_mean(self.loss) # average across batch

    # Predictions
    self.y_pred, self.y_true = self.predict(self.labels, self.logits)

    # Optimize
    self.optimize = self.optimize_step(self.cost,self.global_step)

  def pair_wise_matching(self, rnn_h):
    """
    Args:
      rnn_h: rnn hidden states over time (output of dynamic encoder)
    """
    # Since rnn_h is [batch_size, time, h_size], transpose 2 and 1 dim
    x = tf.transpose(rnn_h, perm=[0, 2, 1])
    # Output of matmul should be [batch_size, time,time]
    p_w = tf.matmul(x,rnn_h)
    return p_w

  def attn_matrices(self, p_w, input_len, batch_size):
    """
    Create column-wise and row-wise softmax, masking 0
    Based on https://arxiv.org/abs/1607.04423
    """
    dims = tf.shape(p_w)[2]
    ones = np.ones([dims,dims])
    ones[input_len[:,None] <= np.arange(d.shape[1])] = 0
    r = np.expand_dims(p,1)

  def embedded(self, word_ids, embedding_tensor, scope="embedding"):
    """Swap ints for dense embeddings, on cpu.
    word_ids correspond the proper row index of the embedding_tensor

    Args:
      words_ids: array of [batch_size x sequence of word ids]
      embedding_tensor: tensor from which to retrieve the embedding, word id
        takes corresponding tensor row
    Returns:
      tensor of shape [batch_size, sequence length, embedding size]
    """
    with tf.variable_scope(scope):
      with tf.device("/cpu:0"):
        inputs = tf.nn.embedding_lookup(embedding_tensor, word_ids)
    return inputs

  def embedding_setup(self, embedding, emb_trainable):
    """ If trainable, returns variable, otherwise the original embedding """
    if emb_trainable == True:
      emb_variable = tf.get_variable(
          name="embedding_matrix", shape=embedding.shape,
          initializer = tf.constant_initializer(embedding))
      return emb_variable
    else:
      return embedding

  def build_cell(self, cell_type="LSTMCell"):
      Cell = locate("tensorflow.contrib.rnn." + cell_type)
      if Cell is None:
        raise ValueError("Invalid cell type " + cell_type)
      cell_fw = Cell(hp.cell_units)
      cell_bw = Cell(hp.cell_units)

      return cell_fw, cell_bw

  def encoder_bi(self, cell_fw, cell_bw, x, seq_len, init_state_fw=None,
                  init_state_bw=None):
    """
    Dynamic bidirectional encoder
    Args:
      cell_fw: forward cell
      cell_bw: backward cell
      x: inputs to encode
      seq_len : length of each row in x batch tensor, needed for dynamic_rnn
    Returns:
      outputs: Tensor, result of the concatenation of
        tuple(output_fw, output_bw) of shape [batch,time,units]
      state: tuple(output_state_fw, output_state_bw) containing the forward
             and the backward final states of bidirectional rnlast hidden state
    """
    # Output is the outputs at all time steps, state is the last state
    with tf.variable_scope("bidirectional_dynamic_rnn"):
      outputs, state = tf.nn.bidirectional_dynamic_rnn(\
                  cell_fw=cell_fw,
                  cell_bw=cell_bw,
                  inputs=x,
                  sequence_length=seq_len,
                  initial_state_fw=init_state_fw,
                  initial_state_bw=init_state_bw,
                  dtype=self.floatX)
      # outputs: a tuple(output_fw, output_bw), all sequence hidden states,
      # each as tensor of shape [batch,time,units]
      # Since we don't need the outputs separate, we concat here
      outputs = tf.concat(outputs,2)
      outputs.set_shape([None, None, self.bi_encoder_hidden])
      # If LSTM cell, then "state" is not a tuple of Tensors but an
      # LSTMStateTuple of "c" and "h". Need to concat separately then new
      if "LSTMStateTuple" in str(type(state[0])):
        c = tf.concat([state[0][0],state[1][0]],axis=1)
        h = tf.concat([state[0][1],state[1][1]],axis=1)
        state = tf.contrib.rnn.LSTMStateTuple(c,h)
      else:
        state = tf.concat(state,1)
        # Manually set shape to Tensor or all hell breaks loose
        state.set_shape([None, self.bi_encoder_hidden])
    return outputs, state

  def optimize_step(self, loss, glbl_step):
    """ Locate optimizer from hparams, take a step """
    Opt = locate("tensorflow.train." + hparams.optimizer)
    if Opt is None:
      raise ValueError("Invalid optimizer: " + hparams.optimizer)
    optimizer = Opt(hparams.l_rate)
    grads_vars = optimizer.compute_gradients(loss)
    capped_grads = [(None if grad is None else tf.clip_by_value(grad, -1., 1.), var)\
                                                  for grad, var in grads_vars]
    take_step = optimizer.apply_gradients(capped_grads, global_step=glbl_step)
    return take_step

  def classification_loss(self, classes_true, classes_logits):
    """ Class loss. If binary, two outputs"""
    entropy_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

    classes_max = tf.argmax(classes_true, axis=1)
    class_loss = entropy_fn(
                      labels=classes_max,
                      logits=classes_logits)
    return class_loss

  def predict(self, labels, logits):
    """ Returns class label (int) for prediction and gold
    Args:
      pred_logits : predicted logits, not yet softmax
      classes : labels as one-hot vectors
    """
    y_pred = tf.nn.softmax(logits)
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.argmax(labels, axis=1)

    return y_pred, y_true

def dense(self, x, in_dim, out_dim, scope, act=None):
  """ Fully connected layer builder"""
  with tf.variable_scope(scope):
    weights = tf.get_variable("weights", shape=[in_dim, out_dim],
              dtype=tf.float32, initializer=glorot())
    biases = tf.get_variable("biases", out_dim,
              dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    # Pre activation
    h = tf.matmul(x,weights) + biases
    # Post activation
    if act:
      h = act(h)
    return h

def seq_length(sequence):
  """ Compute length on the fly in the graph, sequence is [batch, time, dim]"""
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length

