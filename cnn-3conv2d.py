# -*- coding=utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'MNIST_data', 'File path of data!')
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
x = tf.placeholder(tf.float32, [None, 28 * 28])
x_image = tf.reshape(x, [-1, 28, 28, 1])
y_image = tf.placeholder(tf.float32, [None, 10])


def weight_variable(shape):
    return tf.Variable(tf.random_normal(shape, 0, 0.1, tf.float32))


def bias_variable(shape):
    return tf.Variable(tf.random_normal(shape, 0, 0.1, tf.float32))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, [1, 1, 1, 1], "SAME")


def max_pool(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


# 第1层 输入 ?*28*28*1 conv ?*28*28*6 max: ?*14*14*6
W_conv1 = weight_variable([5, 5, 1, 6])
b_conv1 = bias_variable([6])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
max_pool1 = max_pool(h_conv1)
# 第2层 输入 ?*14*14*6 conv ?*14*14*16 max: ?*7*7*16
W_conv2 = weight_variable([5, 5, 6, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(max_pool1, W_conv2) + b_conv2)
max_pool2 = max_pool(h_conv2)
# 第3层 输入 ?*4*4*16 输出120
W_dense3 = weight_variable([7 * 7 * 16, 120])
b_dense3 = bias_variable([120])
max_pool2_flat = tf.reshape(max_pool2, [-1, 7 * 7 * 16])
h_dense3 = tf.nn.relu(tf.matmul(max_pool2_flat, W_dense3) + b_dense3)
keep_prob = tf.placeholder(tf.float32)
h_fc_drop3 = tf.nn.dropout(h_dense3, keep_prob)

# 第4层 输入 120 输出 84
W_dense4 = weight_variable([120, 84])
b_dense4 = bias_variable([84])
h_dense4 = tf.nn.relu(tf.matmul(h_fc_drop3, W_dense4) + b_dense4)

# 第5层 输入 84 输出 10
W_dense5 = weight_variable([84, 10])
b_dense5 = bias_variable([10])
h_dense5 = tf.nn.softmax(tf.matmul(h_dense4, W_dense5) + b_dense5)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_image * tf.log(h_dense5), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(1e-4)
train = optimizer.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_image, 1), tf.argmax(h_dense5, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy, loss = sess.run([accuracy, cross_entropy], feed_dict={x: batch_xs, y_image: batch_ys, keep_prob: 1})
        print('step %d, accuracy: %g, loss: %g' % (i, train_accuracy, loss))
    sess.run([train], feed_dict={x: batch_xs, y_image: batch_ys, keep_prob: 0.5})
print('test accuracy %g'% accuracy.eval(feed_dict={x: mnist.test.images, y_image: mnist.test.labels, keep_prob: 1}))