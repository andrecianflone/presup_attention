
Base settings:
```python
  # General hyper params
  hp = HParams(
    emb_trainable = False,
    batch_size    = 64,
    max_seq_len   = 60,
    max_epochs    = 20,
    early_stop    = 10,
    keep_prob     = 0.5,
    eval_every    = 300,
    num_classes   = 2,
    l_rate        = 0.001,
    cell_units    = 128,
    cell_type     = 'LSTMCell',
    optimizer     = 'AdamOptimizer'
  )

  # Hyper params for dense layers
  hp.update(
    dense_units = 64
  )
  # Hyper params for convnet
  hp.update(
    filt_height  = 3,
    filt_width   = 3,
    h_layers     = 0,
    h_units = self.dense_units,
    conv_strides = [1,2,2,1], #since input is "NHWC", no batch/channel stride
    padding      = "VALID",
    out_channels = 32
  )
```

## Results

### Dataset: WSJ
Model    | param      | value | acc   | epoch

### Dataset: Giga also on val
Base results: 75.31 on epoch 1
Model    | param         | value | acc   | epoch
---------|---------------|-------|-------|
ConvAttn | RNN units     | 256   | 76.01 | 1
ConvAttn | RNN units     | 512   | 69.51 | 3
ConvAttn | batch_norm    | yes   | 50.66 | 1
ConvAttn | h_layers      | 1     | 75.60 | 2
ConvAttn | fine tune emb | no    | 77.48 | 7
ConvAttn | fine tune emb | no    | 77.77 | 6
         | RNN units     | 256   |       |


### Dataset: Giga also on test
Base results: val 78.64, test 78.14, epoch 7
Model     | param         | value | val   | test  | epoch
----------|---------------|-------|-------|
ConvAttn  | RNN units     | 256   | 79.14 | 78.26 | 4
-ConvAttn  | RNN units     | 512   |       |       |
ConvAttn  | batch_norm    | yes   |       |       |
ConvAttn  | h_layers      | 1     |       |       |
ConvAttn  | fine tune emb | no    |       |       |


