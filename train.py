import tensorflow as tf
from model import Attn

def get_batches(tensors):
  batches = batch(
    tensors,
    batch_size=hp.batch_size,
    num_threads=1,
    capacity=32,
    enqueue_many=False,
    shapes=None,
    dynamic_pad=False,
    allow_smaller_final_batch=False,
    shared_name=None,
    name=None
  )

def train_model(params,emb,trX,trY,teX,teY):
  global hp
  hp = params
  with tf.Graph().as_default(), tf.Session() as sess:
    tf.set_random_seed(1)
    model = Attn(hp, emb)
    tf.global_variables_initializer().run()
    for batches in get_batches([trX,trY]):
      fetch = [model.encoded_outputs]
