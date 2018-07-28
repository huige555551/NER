# -*- coding=utf-8 -*-
""""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

flag = tf.app.flags
tf.app.flags.DEFINE_string('data_dir','MNIST_data', 'File path of data!')
mnist = input_data.read_data_sets(tf.app.flags.FLAGS.data_dir, one_hot=True)

x = tf.placeholder(shape=[None, 28*28], dtype=tf.float32)
y = tf.placeholder(shape=[None, 10], dtype=tf.float32)
# W = tf.Variable(tf.zeros([784, 10]))
W = tf.Variable(tf.zeros(shape=[28*28, 10]))
b = tf.Variable(tf.zeros(shape=[10]))
a = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(a), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(0.0001)
train = optimizer.minimize(cross_entropy)

# Test trained model
correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train
sess = tf.InteractiveSession()      # 建立交互式会话
tf.initialize_all_variables().run()
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train.run({x: batch_xs, y: batch_ys})
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))