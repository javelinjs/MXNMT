import os
import mxnet as mx

# path
source_root = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
data_root = os.path.join(source_root, 'RECG')
model_root = os.path.join(source_root, 'RECG', 'model')

if not os.path.exists(model_root):
    os.makedirs(model_root)

# dictionary
bos_word = '<s>'
eos_word = '</s>'
unk_word = '<unk>'
special_words = {unk_word: 1, bos_word: 2, eos_word: 3}
source_vocab_path = os.path.join(data_root, 'vocab.pkl')

# data set
train_source = os.path.join(data_root, 'src.txt')
train_target = os.path.join(data_root, 'target.txt')
train_max_samples = 100000
label_size = 4

# model parameter
batch_size = 128
bucket_stride = 10
buckets = []
for i in range(10, 70, bucket_stride):
    for j in range(10, 70, bucket_stride):
        buckets.append((i, j))
num_hidden = 512  # hidden unit in LSTM cell
num_embed = 512  # embedding dimension
num_lstm_layer = 1  # number of lstm layer

# training parameter
num_epoch = 60
learning_rate = 1
momentum = 0.1
dropout = 0.5
show_every_x_batch = 100
eval_per_x_batch = 400
eval_start_epoch = 4

# model save option
model_save_name = os.path.join(model_root, "recognize")
model_save_freq = 1  # every x epoch
checkpoint_name = os.path.join(model_root, 'checkpoint_model')
checkpoint_freq_batch = 1000  # save checkpoint model every x batch

# train device
train_device = [mx.context.cpu(0)]
# test device
test_device = mx.context.cpu(0)
