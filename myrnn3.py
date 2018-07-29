# -*- coding=utf-8 -*-
import warnings
warnings.filterwarnings('ignore')  # 不打印 warning

import tensorflow as tf

import numpy as np

# 用tensorflow 导入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 看看咱们样本的数量
print(mnist.test.labels.shape)
print(mnist.train.labels.shape)

lr = 1e-3
input_size = 28      # 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
timestep_size = 28   # 时序持续长度为28，即每做一次预测，需要先输入28行
hidden_size = 256    # 隐含层的数量
layer_num = 2        # LSTM layer 的层数
class_num = 10       # 最后输出分类类别数量，如果是回归预测的话应该是 1
cell_type = "lstm"   # lstm 或者 block_lstm

X_input = tf.placeholder(tf.float32, [None, 784])
y_input = tf.placeholder(tf.float32, [None, class_num])
# 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32, batch_size = 128
keep_prob = tf.placeholder(tf.float32, [])

X = tf.reshape(X_input, [-1, 28, 28])

# ** 步骤2：创建 lstm 结构
def lstm_cell(num_nodes, keep_prob):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_nodes)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    return lstm_cell

mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(hidden_size, keep_prob) for _ in range(layer_num)], state_is_tuple = True)

# **步骤3：用全零来初始化state
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

outputs = list()
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        (cell_output, state) = mlstm_cell(X[:, timestep, :],state)
        outputs.append(cell_output)
h_state = outputs[-1]