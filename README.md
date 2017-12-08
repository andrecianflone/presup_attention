
## Task
Given a sample text the model must predict if it contains a presupposition triggering. Pressupositions are triggered by keywords such as "also", "again". The keywords are removed since their position make the task easier.

## Models
We experiment with 3 models for presupposition triggering.

### 1. PairWiseAttn
The base model for all others. The input is encoded via an RNN.
We take all output states for a sample and compute matrix multiplication on itself,
giving us a pairwise matching score across all input pairs. We then produce two
matrices, one row softmax and a column softmax.

### 2. Attention-over-Attention
Building on the base model, we compute a column-wise average over the row-wise softmax matrix, giving us vector `c`. We compute the dot product of vector `c` with the column-wise softmax matrix, giving us a sum-attention vector `att-o-att`.
This is based on [Cui et al, 2017](https://arxiv.org/pdf/1607.04423.pdf)'s Attention-over-Attention model for cloze-style reading comprehension.
However, we do not implement the last step where a a word is predicted by summing over `att-o-att` vector. Our final layer is a fully connected layer between `c` and our two classes.

### 3. Convolution-over-Attention
Given the two normalized pair-wise matching score matrices, we convolve over these.
The intuition is to locate attention groupings which seems supported by qualitative analysis of our data.

Base settings:
```shell
'batch_norm'            : False,
'batch_size'            : 64,
'cell_type'             : 'LSTMCell',
'cell_units'            : 128,
'conv_strides'          : [1, 2, 2, 1],
'data_dir'              : '../presup_giga_also/',
'early_stop'            : 10,
'emb_trainable'         : False,
'eval_every'            : 300,
'fc_units'              : 64,
'filt_height'           : 3,
'filt_width'            : 3,
'h_layers'              : 0,
'keep_prob'             : 0.5,
'l_rate'                : 0.001,
'max_epochs'            : 20,
'max_seq_len'           : 60,
'model'                 : 'AttnAttn',
'num_classes'           : 2,
'optimizer'             : 'AdamOptimizer',
'out_channels'          : 32,
'padding'               : 'VALID',
'rnn_in_keep_prob'      : 1.0,
'variational_recurrent' : False,
'word_gate'             : False


## Results

### Dataset: Giga also
Single param variation
Model       | param      | value  | command | val acc | test acc | epoch
------------|------------|--------|---------|---------|----------|
AttnAttn    | *base*     | *base* |         | 79.30   | 78.90    | 5
AttnAttn    | word gate  | True   |         | 79.51   | 78.94    | 5
biRNN only  | *base*     | *base* | 3       | 81.64   | 80.87    | 4
RNN only    | cell_units | 300    | 4       | 83.15   | 81.16    | 34
AttnAttnSum | cell_units | 300    | 5       | 82.80   | 81.45    | 16

Multi param
Model    | param/value                                                            | command | val acc | test acc | epoch
---------|------------------------------------------------------------------------|---------|---------|----------|
AttnAttn | cell units/256, in keep prob/0.5, word gate/True                       | 1       | 81.43   | 80.12    | 17
AttnAttn | cell units/256, in keep prob/0.5, word gate/True, emb_trainable/true   | 2       | 83.48   | 70.06    | 2
AttnAttn | cell units/256, in keep prob/0.5, word gate/True , same random for unk | 1       | 80.08   | 78.20    | 23

Command:
1: python main.py --cell_units 256 --rnn_in_keep_prob 0.5 --word_gate --ckpt_name also_word_fix_proc
2: python main.py --cell_units 256 --rnn_in_keep_prob 0.5 --word_gate --emb_trainable True
3: python main.py --model NoAttn --ckpt_name also_noattn
4: python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --model RNN_base --ckpt_name also_uni_rnn
5: python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --model AttnAttnSum --ckpt_name also_uni_attnattnsum

#### OLD
AttnAttn  | RNN units       | 256    | 78.79 | 78.68 | 3
AttnAttn  | RNN units       | 512    | 78.76 | 78.57 | 5
AttnAttn  | h_layers        | 1      | 76.32 | 75.86 | 8
AttnAttn  | input keep rate | 0.5    | 78.28 | 77.87 | 15

Multi param
Model      | param           | value | val   | test  | epoch
-----------|-----------------|-------|-------|-------|
AttnAttn   | input keep rate | 0.5   | 78.88 | 78.28 | 22
           | var recurrent   | True  |       |       |
AttnAttn   | input keep rate | 0.5   | 78.79 | 78.08 |
           | var recurrent   | True  |       |       |
           | RNN units       | 256   |       |       |

### Dataset: Giga also
Model    | param         | value  | val   | test  | epoch
---------|---------------|--------|-------|-------|---
ConvAttn | *base*        | *base* | 78.64 | 78.14 | 7
ConvAttn | RNN units     | 256    | 79.14 | 78.26 | 4
ConvAttn | RNN units     | 512    | 75.12 | 74.27 |
ConvAttn | batch_norm    | yes    | 49.97 | 49.95 | 1
ConvAttn | h_layers      | 1      | 78.30 | 77.72 | 8
ConvAttn | fine tune emb | yes    | 79.12 | 73.19 |


