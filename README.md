

# Sample run
Simple example to run ptb dataset using attention model, no saving
```
python main.py --data_dir /home/rldata/new_presup_data/wsj_natural/ --pickle /home/ml/acianf/projects/presup/presup_wsj/processed.pkl --model AttnAttnSum --eval_every 20 --batch_size 32
python main.py --data_dir /home/rldata/presup_data/presup_wsj/natural/ --pickle ../presup_wsj/processed.pkl --model AttnAttnSum --eval_every 20 --batch_size 32
```

# New experiments
## Datasets:
Following datasets are used. First line is root directory which contains train/valid/test folders. Second line is the "processed.pkl" file.

WSJ Natural with balanced training
/home/rldata/new_presup_data/wsj_natural_bal_train/
/home/rldata/new_presup_data/wsj_balanced/all/processed.pkl

Giga again
/home/rldata/new_presup_data/giga_individual/again/
/home/rldata/new_presup_data/giga_individual/again/train/processed.pkl

Giga still
/home/rldata/new_presup_data/giga_individual/still/
/home/rldata/new_presup_data/giga_individual/still/train/processed.pkl

Giga too
/home/rldata/new_presup_data/giga_individual/too/
/home/rldata/new_presup_data/giga_individual/too/train/processed.pkl

Giga yet
/home/rldata/new_presup_data/giga_individual/yet/
/home/rldata/new_presup_data/giga_individual/yet/train/processed.pkl

Giga all balanced
/home/rldata/new_presup_data/giga_all_balanced/
/home/rldata/new_presup_data/giga_all_balanced/train/processed.pkl

```
############################
# ATTN W/ POS
############################
# WSJ natural distribution with balanced train
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --postags --eval_every 32 --model AttnAttnSum --ckpt_name wsj_natural --data_dir /home/rldata/new_presup_data/wsj_natural_bal_train/ --pickle /home/rldata/new_presup_data/wsj_balanced/all/processed.pkl

# Giga again
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --postags --eval_every 1000 --model AttnAttnSum --ckpt_name giga_again --data_dir /home/rldata/new_presup_data/giga_individual/again/ --pickle /home/rldata/new_presup_data/giga_individual/again/train/processed.pkl

# Giga still
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --postags --eval_every 2000 --model AttnAttnSum --ckpt_name giga_still --data_dir /home/rldata/new_presup_data/giga_individual/still/ --pickle /home/rldata/new_presup_data/giga_individual/still/train/processed.pkl

# Giga too
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --postags --eval_every 1000 --model AttnAttnSum --ckpt_name giga_too_attn_pos --data_dir /home/rldata/new_presup_data/giga_individual/too/ --pickle /home/rldata/new_presup_data/giga_individual/too/train/processed.pkl

# Giga yet
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --postags --eval_every 1000 --model AttnAttnSum --ckpt_name giga_yet_attn_pos --data_dir /home/rldata/new_presup_data/giga_individual/yet/ --pickle /home/rldata/new_presup_data/giga_individual/yet/train/processed.pkl

# Giga all balanced
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --postags --eval_every 7000 --model AttnAttnSum --ckpt_name giga_all_attn_pos --data_dir /home/rldata/new_presup_data/giga_all_balanced/ --pickle /home/rldata/new_presup_data/giga_all_balanced/train/processed.pkl
############################
# ATTN NO POS
############################
# WSJ natural distribution with balanced train
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --eval_every 32 --model AttnAttnSum --ckpt_name wsj_natural_attn_nopos --data_dir /home/rldata/new_presup_data/wsj_natural_bal_train/ --pickle /home/rldata/new_presup_data/wsj_balanced/all/processed.pkl

# Giga again
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --eval_every 1000 --model AttnAttnSum --ckpt_name giga_again_attn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/again/ --pickle /home/rldata/new_presup_data/giga_individual/again/train/processed.pkl

# Giga still
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --eval_every 2000 --model AttnAttnSum --ckpt_name giga_still_nopos --data_dir /home/rldata/new_presup_data/giga_individual/still/ --pickle /home/rldata/new_presup_data/giga_individual/still/train/processed.pkl

# Giga too
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --eval_every 1000 --model AttnAttnSum --ckpt_name giga_too_attn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/too/ --pickle /home/rldata/new_presup_data/giga_individual/too/train/processed.pkl

# Giga yet
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --eval_every 1000 --model AttnAttnSum --ckpt_name giga_yet_attn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/yet/ --pickle /home/rldata/new_presup_data/giga_individual/yet/train/processed.pkl

# Giga all balanced
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --eval_every 7000 --model AttnAttnSum --ckpt_name giga_all_attn_nopos --data_dir /home/rldata/new_presup_data/giga_all_balanced/ --pickle /home/rldata/new_presup_data/giga_all_balanced/train/processed.pkl
############################
# RNN W/ POS
############################
# WSJ natural distribution with balanced train
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --postags --eval_every 32 --model RNN_base --ckpt_name wsj_natural_rnn_pos --data_dir /home/rldata/new_presup_data/wsj_natural_bal_train/ --pickle /home/rldata/new_presup_data/wsj_balanced/all/processed.pkl

# Giga again
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --postags --eval_every 1000 --model RNN_base  --ckpt_name giga_again_rnn_pos --data_dir /home/rldata/new_presup_data/giga_individual/again/ --pickle /home/rldata/new_presup_data/giga_individual/again/train/processed.pkl

# Giga still
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --postags --eval_every 2000 --model RNN_base --ckpt_name giga_still_rnn_pos --data_dir /home/rldata/new_presup_data/giga_individual/still/ --pickle /home/rldata/new_presup_data/giga_individual/still/train/processed.pkl

# Giga too
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --postags --eval_every 1000 --model RNN_base --ckpt_name giga_too_rnn_pos --data_dir /home/rldata/new_presup_data/giga_individual/too/ --pickle /home/rldata/new_presup_data/giga_individual/too/train/processed.pkl

# Giga yet
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --postags --eval_every 1000 --model RNN_base --ckpt_name giga_yet_rnn_pos --data_dir /home/rldata/new_presup_data/giga_individual/yet/ --pickle /home/rldata/new_presup_data/giga_individual/yet/train/processed.pkl

# Giga all balanced
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --postags --eval_every 7000 --model RNN_base  --ckpt_name giga_all_rnn_pos --data_dir /home/rldata/new_presup_data/giga_all_balanced/ --pickle /home/rldata/new_presup_data/giga_all_balanced/train/processed.pkl
############################
# RNN NO POS
############################
# WSJ natural distribution with balanced train
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --eval_every 32 --model RNN_base --ckpt_name wsj_natural_rnn_nopos --data_dir /home/rldata/new_presup_data/wsj_natural_bal_train/ --pickle /home/rldata/new_presup_data/wsj_balanced/all/processed.pkl

# Giga again
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --eval_every 1000 --model RNN_base --ckpt_name giga_again_rnn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/again/ --pickle /home/rldata/new_presup_data/giga_individual/again/train/processed.pkl

# Giga still
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --eval_every 2000 --model RNN_base --ckpt_name giga_still_rnn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/still/ --pickle /home/rldata/new_presup_data/giga_individual/still/train/processed.pkl

# Giga too
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --eval_every 1000 --model RNN_base --ckpt_name giga_too_rnn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/too/ --pickle /home/rldata/new_presup_data/giga_individual/too/train/processed.pkl

# Giga yet
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --eval_every 1000 --model RNN_base --ckpt_name giga_yet_rnn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/yet/ --pickle /home/rldata/new_presup_data/giga_individual/yet/train/processed.pkl

# Giga all balanced
python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --eval_every 7000 --model RNN_base  --ckpt_name giga_all_rnn_nopos --data_dir /home/rldata/new_presup_data/giga_all_balanced/ --pickle /home/rldata/new_presup_data/giga_all_balanced/train/processed.pkl


############################
# CNN W/ POS
############################
# WSJ natural distribution with balanced train
python main.py --postags --eval_every 32 --model CNN --ckpt_name wsj_natural_cnn_pos --data_dir /home/rldata/new_presup_data/wsj_natural_bal_train/ --pickle /home/rldata/new_presup_data/wsj_balanced/all/processed.pkl

# Giga again
python main.py  --postags --eval_every 1000 --model CNN --ckpt_name giga_again_cnn_pos --data_dir /home/rldata/new_presup_data/giga_individual/again/ --pickle /home/rldata/new_presup_data/giga_individual/again/train/processed.pkl

# Giga still
python main.py  --postags --eval_every 2000 --model CNN --ckpt_name giga_still_cnn_pos --data_dir /home/rldata/new_presup_data/giga_individual/still/ --pickle /home/rldata/new_presup_data/giga_individual/still/train/processed.pkl

# Giga too
python main.py  --postags --eval_every 1000 --model CNN --ckpt_name giga_too_cnn_pos --data_dir /home/rldata/new_presup_data/giga_individual/too/ --pickle /home/rldata/new_presup_data/giga_individual/too/train/processed.pkl

# Giga yet
python main.py  --postags --eval_every 1000 --model CNN --ckpt_name giga_yet_cnn_pos --data_dir /home/rldata/new_presup_data/giga_individual/yet/ --pickle /home/rldata/new_presup_data/giga_individual/yet/train/processed.pkl

# Giga all balanced
python main.py  --postags --eval_every 7000 --model CNN --ckpt_name giga_all_cnn_pos --data_dir /home/rldata/new_presup_data/giga_all_balanced/ --pickle /home/rldata/new_presup_data/giga_all_balanced/train/processed.pkl

############################
# CNN NO POS
############################

# WSJ natural distribution with balanced train
python main.py  --eval_every 32   --model CNN --ckpt_name wsj_natural_cnn_nopos --data_dir /home/rldata/new_presup_data/wsj_natural_bal_train/ --pickle /home/rldata/new_presup_data/wsj_balanced/all/processed.pkl

# Giga again
python main.py  --eval_every 1000 --model CNN --ckpt_name giga_again_cnn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/again/ --pickle /home/rldata/new_presup_data/giga_individual/again/train/processed.pkl

# Giga still
python main.py  --eval_every 2000 --model CNN --ckpt_name giga_still_cnn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/still/ --pickle /home/rldata/new_presup_data/giga_individual/still/train/processed.pkl

# Giga too
python main.py  --eval_every 1000 --model CNN --ckpt_name giga_too_cnn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/too/ --pickle /home/rldata/new_presup_data/giga_individual/too/train/processed.pkl

# Giga yet
python main.py  --eval_every 1000 --model CNN --ckpt_name giga_yet_cnn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/yet/ --pickle /home/rldata/new_presup_data/giga_individual/yet/train/processed.pkl

# Giga all balanced
python main.py  --eval_every 7000 --model CNN --ckpt_name giga_all_cnn_nopos --data_dir /home/rldata/new_presup_data/giga_all_balanced/ --pickle /home/rldata/new_presup_data/giga_all_balanced/train/processed.pkl


############################
# Load saved model
############################

python main.py --ckpt_name wsj_natural --data_dir /home/rldata/new_presup_data/wsj_natural_bal_train/ --pickle /home/rldata/new_presup_data/wsj_balanced/all/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_again --data_dir /home/rldata/new_presup_data/giga_individual/again/ --pickle /home/rldata/new_presup_data/giga_individual/again/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_still --data_dir /home/rldata/new_presup_data/giga_individual/still/ --pickle /home/rldata/new_presup_data/giga_individual/still/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_too_attn_pos --data_dir /home/rldata/new_presup_data/giga_individual/too/ --pickle /home/rldata/new_presup_data/giga_individual/too/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_yet_attn_pos --data_dir /home/rldata/new_presup_data/giga_individual/yet/ --pickle /home/rldata/new_presup_data/giga_individual/yet/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_all_attn_pos --data_dir /home/rldata/new_presup_data/giga_all_balanced/ --pickle /home/rldata/new_presup_data/giga_all_balanced/train/processed.pkl --load_saved --mode 0

python main.py --ckpt_name wsj_natural_attn_nopos --data_dir /home/rldata/new_presup_data/wsj_natural_bal_train/ --pickle /home/rldata/new_presup_data/wsj_balanced/all/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_again_attn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/again/ --pickle /home/rldata/new_presup_data/giga_individual/again/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_still_attn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/still/ --pickle /home/rldata/new_presup_data/giga_individual/still/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_too_attn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/too/ --pickle /home/rldata/new_presup_data/giga_individual/too/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_yet_attn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/yet/ --pickle /home/rldata/new_presup_data/giga_individual/yet/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_all_attn_nopos --data_dir /home/rldata/new_presup_data/giga_all_balanced/ --pickle /home/rldata/new_presup_data/giga_all_balanced/train/processed.pkl --load_saved --mode 0

python main.py --ckpt_name wsj_natural_rnn_pos --data_dir /home/rldata/new_presup_data/wsj_natural_bal_train/ --pickle /home/rldata/new_presup_data/wsj_balanced/all/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_again_rnn_pos --data_dir /home/rldata/new_presup_data/giga_individual/again/ --pickle /home/rldata/new_presup_data/giga_individual/again/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_still_rnn_pos --data_dir /home/rldata/new_presup_data/giga_individual/still/ --pickle /home/rldata/new_presup_data/giga_individual/still/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_too_rnn_pos --data_dir /home/rldata/new_presup_data/giga_individual/too/ --pickle /home/rldata/new_presup_data/giga_individual/too/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_yet_rnn_pos --data_dir /home/rldata/new_presup_data/giga_individual/yet/ --pickle /home/rldata/new_presup_data/giga_individual/yet/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_all_rnn_pos --data_dir /home/rldata/new_presup_data/giga_all_balanced/ --pickle /home/rldata/new_presup_data/giga_all_balanced/train/processed.pkl --load_saved --mode 0

python main.py --ckpt_name wsj_natural_rnn_nopos --data_dir /home/rldata/new_presup_data/wsj_natural_bal_train/ --pickle /home/rldata/new_presup_data/wsj_balanced/all/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_again_rnn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/again/ --pickle /home/rldata/new_presup_data/giga_individual/again/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_still_rnn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/still/ --pickle /home/rldata/new_presup_data/giga_individual/still/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_too_rnn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/too/ --pickle /home/rldata/new_presup_data/giga_individual/too/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_yet_rnn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/yet/ --pickle /home/rldata/new_presup_data/giga_individual/yet/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_all_rnn_nopos --data_dir /home/rldata/new_presup_data/giga_all_balanced/ --pickle /home/rldata/new_presup_data/giga_all_balanced/train/processed.pkl --load_saved --mode 0


python main.py --ckpt_name wsj_natural_cnn_pos --data_dir /home/rldata/new_presup_data/wsj_natural_bal_train/ --pickle /home/rldata/new_presup_data/wsj_balanced/all/processed.pkl --load_saved --mode 0

python main.py --ckpt_name giga_again_cnn_pos --data_dir /home/rldata/new_presup_data/giga_individual/again/ --pickle /home/rldata/new_presup_data/giga_individual/again/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_still_cnn_pos --data_dir /home/rldata/new_presup_data/giga_individual/still/ --pickle /home/rldata/new_presup_data/giga_individual/still/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_too_cnn_pos --data_dir /home/rldata/new_presup_data/giga_individual/too/ --pickle /home/rldata/new_presup_data/giga_individual/too/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_yet_cnn_pos --data_dir /home/rldata/new_presup_data/giga_individual/yet/ --pickle /home/rldata/new_presup_data/giga_individual/yet/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_all_cnn_pos --data_dir /home/rldata/new_presup_data/giga_all_balanced/ --pickle /home/rldata/new_presup_data/giga_all_balanced/train/processed.pkl --load_saved --mode 0

python main.py --ckpt_name wsj_natural_cnn_nopos --data_dir /home/rldata/new_presup_data/wsj_natural_bal_train/ --pickle /home/rldata/new_presup_data/wsj_balanced/all/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_again_cnn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/again/ --pickle /home/rldata/new_presup_data/giga_individual/again/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_still_cnn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/still/ --pickle /home/rldata/new_presup_data/giga_individual/still/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_too_cnn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/too/ --pickle /home/rldata/new_presup_data/giga_individual/too/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_yet_cnn_nopos --data_dir /home/rldata/new_presup_data/giga_individual/yet/ --pickle /home/rldata/new_presup_data/giga_individual/yet/train/processed.pkl --load_saved --mode 0
python main.py --ckpt_name giga_all_cnn_nopos --data_dir /home/rldata/new_presup_data/giga_all_balanced/ --pickle /home/rldata/new_presup_data/giga_all_balanced/train/processed.pkl --load_saved --mode 0


```

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

### Dataset: PTB
Multi param
Model       | param/value               | command | val acc | test acc | epoch
------------|---------------------------|---------|---------|----------|
RNN only    | cell_units 64, unidirect  | 1       | 78.93   | 66.94    | 9
RNN only    | cell_units 128, unidirect | 2       | 80.31   | 71.93    | 12
RNN only    | cell_units 200, unidirect | 3       | 80.69   | 73.18    | 9
RNN only    | cell_units 300, unidirect | 4       | 81.27   | 73.18    | 11
RNN only    | cell_units 150, bidirect  | 5       | 80.69   | 68.61    | 14
AttnAttnSum | cell_units, 300           | 6       | 81.47  | 75.05   | 18
RNN only 72.56, 73.39
AttnAttnSum 71.93, 71.73, 72.77

Command:
1: python main.py --data_dir ../presup_wsj/ --model RNN_base  --batch_size 32 --eval_every 20 --cell_units 64
2: python  main.py --data_dir ../presup_wsj/ --model RNN_base  --batch_size 32 --eval_every 20 --cell_units 128
3: python  main.py --data_dir ../presup_wsj/ --model RNN_base  --batch_size 32 --eval_every 20 --cell_units 200
4: python  main.py --data_dir ../presup_wsj/ --model RNN_base  --batch_size 32 --eval_every 20 --cell_units 300
5: python  main.py --data_dir ../presup_wsj/ --model RNN_base  --batch_size 32 --eval_every 20 --cell_units 150 --birnn
6: python  main.py --data_dir ../presup_wsj/ --model AttnAttnSum  --batch_size 32 --eval_every 20 --cell_units 300

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
Model       | param/value                                                            | command | val acc | test acc | epoch
------------|------------------------------------------------------------------------|---------|---------|----------|
AttnAttn    | cell units/256, in keep prob/0.5, word gate/True                       | 1       | 81.43   | 80.12    | 17
AttnAttn    | cell units/256, in keep prob/0.5, word gate/True, emb_trainable/true   | 2       | 83.48   | 70.06    | 2
AttnAttn    | cell units/256, in keep prob/0.5, word gate/True , same random for unk | 1       | 80.08   | 78.20    | 23
AttnAttnSum | cell units/150, parallel                                               | 6       | 82.56   | 81.26    | 44
AttnAttnSum | birnn, cell_units 150                                                  | 7       | 83.24   | 81.46    | 52
RNN only    | cell_units 300, postags                                                | 8       | 82.94   | 81.81    | 24
AttnAttnSum | birnn, cell_units 300, word gate, pos                                  | 9       | 83.40   | 82.19    | 20

Command:
1: python main.py --cell_units 256 --rnn_in_keep_prob 0.5 --word_gate --ckpt_name also_word_fix_proc
2: python main.py --cell_units 256 --rnn_in_keep_prob 0.5 --word_gate --emb_trainable True
3: python main.py --model NoAttn --ckpt_name also_noattn
4: python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --model RNN_base --ckpt_name also_uni_rnn
5: python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --model AttnAttnSum --ckpt_name also_uni_attnattnsum
6: python  main.py --cell_units 150 --rnn_in_keep_prob 0.5 --model AttnAttnSum --parallel --ckpt_name also_uni_attnattnsum_parallel
7: python main.py --cell_units 150 --birnn --rnn_in_keep_prob 0.5 --model AttnAttnSum --ckpt_name also_bi_attnattnsum
8: python main.py --cell_units 300 --postags --rnn_in_keep_prob 0.5 --model RNN_base --ckpt_name also_uni_rnn_pos
9: python main.py --cell_units 300 --rnn_in_keep_prob 0.5 --birnn --word_gate --postags --model AttnAttnSum --ckpt_name also_bi_300_word_pos_attnattnsum
10: python main.py --cell_units 150 --rnn_in_keep_prob 0.5 --birnn --word_gate --postags --model AttnAttnSum --ckpt_name also_bi_word_pos_attnattnsum

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


