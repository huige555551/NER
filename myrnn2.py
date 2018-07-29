# -*- coding=utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
import warnings

warnings.filterwarnings('ignore')
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'MNIST_data', 'File path!')
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
print (FLAGS.data_dir)
print('test image:', mnist.test.images.shape)
print('test labels:',mnist.test.labels.shape)
print('train image:',mnist.train.images.shape)
print('train labels:',mnist.train.labels.shape)


# super parameters
lr = 1e-3
training_steps = 1000

# netword parameter
input_size = 28
timestep_size = 28
hidden_size = 256
layer_num = 2
class_num = 10

# input

_X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, class_num])
# 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32, batch_size = 128
keep_prob = tf.placeholder(tf.float32, [])

X = tf.reshape(_X, [-1, 28, 28])


def lstm_cell(num_nodes, keep_prob):
    lstm_cell = rnn.BasicLSTMCell(num_nodes)
    lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    return lstm_cell


mlstm_cell = rnn.MultiRNNCell([lstm_cell(hidden_size, keep_prob) for _ in range(layer_num)])

init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

outputs = list()
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
        outputs.append(cell_output)
h_state = outputs[-1]

W = tf.Variable(tf.random_normal([hidden_size, class_num], 0, 0.1, tf.float32))
b = tf.Variable(tf.random_normal([class_num], 0, 0.1, tf.float32))
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + b)

# sess = tf.Session()

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pre), reduction_indices=[1]))  # 损失函数，交叉熵
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 使用adam优化
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pre, 1))  # 计算准确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# sess.run(tf.initialize_all_variables()) # 变量初始化
sess = tf.InteractiveSession()  # 建立交互式会话
tf.initialize_all_variables().run()
# 用于记录每次训练后loss的值
loss_val = []
for i in range(training_steps):
    _batch_size = 50
    batch = mnist.train.next_batch(_batch_size)
    if i % 100 == 0:
        train_accuracy, loss = sess.run([accuracy, cross_entropy], feed_dict={
            _X: batch[0], y: batch[1], keep_prob: 1, batch_size: _batch_size})
        print("step %d, training accuracy %g, loss %.4f" % (i, train_accuracy, loss))
        # print('loss: %.4f' % cross_entropy)
    # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    sess.run([train_step], feed_dict={_X: batch[0], y: batch[1], keep_prob: 1, batch_size: _batch_size})
print(sess.run(accuracy, feed_dict={
    _X: mnist.test.images, y:mnist.test.labels, keep_prob: 1, batch_size: mnist.test.images.shape[0]}))
