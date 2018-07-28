# -*- coding=utf-8 -*-
# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'MNIST_data', 'Directory for storing data') # 第一次启动会下载文本资料，放在/tmp/data文件夹下

print(FLAGS.data_dir)
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

def weight_variable(shape):
    # initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
    initial = tf.random_normal(shape=shape,mean=0, stddev=0.01, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    # initial = tf.constant(0.1, shape=shape)
    initial = tf.random_normal(shape=shape,mean=0, stddev=0.01, dtype=tf.float32)
    return tf.Variable(initial)

def conv2d(x, W):
    """
    tf.nn.conv2d功能：给定4维的input和filter，计算出一个2维的卷积结果
    前几个参数分别是input, filter, strides, padding, use_cudnn_on_gpu, ...
    input   的格式要求为一个张量，[batch, in_height, in_width, in_channels],批次数，图像高度，图像宽度，通道数
    filter  的格式为[filter_height, filter_width, in_channels, out_channels]，滤波器高度，宽度，输入通道数，输出通道数
    strides 一个长为4的list. 表示每次卷积以后在input中滑动的距离
    padding 有SAME和VALID两种选项，表示是否要保留不完全卷积的部分。如果是SAME，则保留
    use_cudnn_on_gpu 是否使用cudnn加速。默认是True
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """
    tf.nn.max_pool 进行最大值池化操作,而avg_pool 则进行平均值池化操作
    几个参数分别是：value, ksize, strides, padding,
    value:  一个4D张量，格式为[batch, height, width, channels]，与conv2d中input格式一样
    ksize:  长为4的list,表示池化窗口的尺寸
    strides: 窗口的滑动值，与conv2d中的一样
    padding: 与conv2d中用法一样。
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1,28,28,1]) #将输入按照 conv2d中input的格式来reshape，reshape

"""
# 第一层
# 卷积核(filter)的尺寸是5*5, 通道数为1，输出通道为6，即feature map 数目为6
# 又因为strides=[1,1,1,1] 所以单个通道的输出尺寸应该跟输入图像一样。即总的卷积输出应该为?*28*28*6
# 也就是单个通道输出为28*28，共有6个通道,共有?个批次
# 在池化阶段，ksize=[1,2,2,1] 那么卷积结果经过池化以后的结果，其尺寸应该是？*14*14*6
"""
W_conv1 = weight_variable([5, 5, 1, 32])  # 卷积是在每个5*5的patch中算出32个特征，分别是patch大小，输入通道数目，输出通道数目
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

"""
# 第二层
# 卷积核5*5，输入通道为6，输出通道为16。
# 卷积前图像的尺寸为 ?*14*14*6， 卷积后为?*14*14*16
# 池化后，输出的图像尺寸为?*7*7*16
"""
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 第三层 是个全连接层,输入维数7*7*64, 输出维数为120
W_fc1 = weight_variable([7 * 7 * 64, 512])
b_fc1 = bias_variable([512])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32) # 这里使用了drop out,即随机安排一些cell输出值为0，可以防止过拟合
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第四层 是个全连接层,输入维数120, 输出维数为84
W_fc2 = weight_variable([512, 84])
b_fc2 = bias_variable([84])
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

# 第五层，输入1024维，输出10维，也就是具体的0~9分类
W_fc3 = weight_variable([84, 10])
b_fc3 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3) # 使用softmax作为多分类激活函数
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])) # 损失函数，交叉熵
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # 使用adam优化
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) # 计算准确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables()) # 变量初始化
# 用于记录每次训练后loss的值
loss_val = []
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy, loss = sess.run([accuracy, cross_entropy], feed_dict={
            x:batch[0], y_: batch[1]})
        print("step %d, training accuracy %g, loss %.4f"%(i, train_accuracy, loss))
        # print('loss: %.4f' % cross_entropy)
    # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    sess.run([train_step],feed_dict={x: batch[0], y_: batch[1]})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels}))