# -*- coding=utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import warnings
import numpy as np
import logging
from myrnn2_config import FLAGS
import matplotlib.pyplot as plt
import myrnn2_utils
import myrnn2_model
import tensorboard

warnings.filterwarnings('ignore')

# class LSTM_net():
#     def __init__(self):
#         self.lr = FLAGS.lr
#         self.training_steps = FLAGS.training_steps
#
#         # netword parameter
#         self.input_size = FLAGS.input_size
#         self.timestep_size = FLAGS.timestep_size
#         self.hidden_size = FLAGS.hidden_size
#         self.layer_num = FLAGS.layer_num
#         self.class_num = FLAGS.class_num
#
#         with tf.variable_scope('Inputs'):
#             self._X = tf.placeholder(tf.float32, [None, 784], name="_X")
#             self.y = tf.placeholder(tf.float32, [None, self.class_num], name="y")
#         # 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
#         self.batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32, batch_size = 128
#         self.keep_prob = tf.placeholder(tf.float32, [])
#         # input [None, timesteps, num_input]
#         self.X = tf.reshape(self._X, [-1, 28, 28])
#         self.outputs = list()
#
#         self._build_net()
#
#     def _build_net(self):
#         self.h_state = self.lstm_net()
#         # self.h_state = self.lstm()
#         self.W = tf.Variable(tf.random_normal([self.hidden_size, self.class_num], 0, 0.1, tf.float32))
#         self.b = tf.Variable(tf.random_normal([self.class_num], 0, 0.1, tf.float32))
#         y_pre = tf.nn.softmax(tf.matmul(self.h_state, self.W) + self.b)
#
#         # Define loss and optimizer
#         self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(y_pre), reduction_indices=[1]))  # 损失函数，交叉熵
#         self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)  # 使用adam优化
#
#         # Evaluate model (with test logits, for dropout to be disabled)
#         self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_pre, 1))  # 计算准确度
#         self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
#         self.saver = tf.train.Saver()
#
#     # lstm_cell和lstm_net可以简化为下面这个函数
#     def lstm(self):
#         x = tf.unstack(self.X, self.timestep_size, 1)
#         lstm_cell = rnn.BasicLSTMCell(self.hidden_size)
#         (outputs, states) = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
#         return outputs[-1]
#
#     def lstm_cell(self):
#         lstm_cell = rnn.BasicLSTMCell(self.hidden_size)
#         lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
#         return lstm_cell
#
#     def lstm_net(self):
#         mlstm_cell = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)])
#         init_state = mlstm_cell.zero_state(self.batch_size, dtype=tf.float32)
#         state = init_state
#         with tf.variable_scope('RNN'):
#             for timestep in range(self.timestep_size):
#                 (cell_output, state) = mlstm_cell(self.X[:, timestep, :], state)
#                 self.outputs.append(cell_output)
#         # outputs.shape = [batch_size, timestep_size, hidden_size]
#         h_state = self.outputs[-1]
#         return h_state


def load_model(sess, ckpt, lstm_net, logger):
    if ckpt is not None:
        path = ckpt.model_checkpoint_path
        logger.info('loading pre-trained model from {}....'.format(path))
        lstm_net.saver.restore(sess, path)
        return int(path[path.find('-') + 1:-5])
    else:
        logger.info('No pre-trained model from {}....'.format(FLAGS.model_path))
        return 0


def train(lstm_net, sess, ckpt):
    logger = myrnn2_utils.get_logger(FLAGS.train_log_file, logging.DEBUG)
    mnist = myrnn2_utils.load_data(FLAGS.data_dir, logger)
    # 加载已经训练好的模型，继续进行训练
    # if ckpt is not None:
    #     path = ckpt.model_checkpoint_path
    #     print 'loading pre-trained model from %s....' % path
    #     saver.restore(sess, path)
    #     trained_times = int(path[path.find('-') + 1:-5])
    trained_times = load_model(sess, ckpt, lstm_net, logger)
    # trained_times = 0
    # tf.summary.scalar('loss', lstm_net.cross_entropy)
    # merged_summary_op = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter('graph', sess.graph)
    for i in range(lstm_net.training_steps - trained_times):
        _batch_size = FLAGS.train_batch_size
        batch = mnist.train.next_batch(_batch_size)
        # summary_str = sess.run(merged_summary_op, feed_dict={
        #         lstm_net._X: batch[0], lstm_net.y: batch[1], lstm_net.keep_prob: FLAGS.test_keep_prob, lstm_net.batch_size: _batch_size})
        # summary_writer.add_summary(summary_str, i)
        if i % 100 == 0:
            train_accuracy, loss = sess.run([lstm_net.accuracy, lstm_net.cross_entropy], feed_dict={
                lstm_net._X: batch[0], lstm_net.y: batch[1], lstm_net.keep_prob: FLAGS.test_keep_prob, lstm_net.batch_size: _batch_size})

            logger.info("step %d, training accuracy %g, loss %.4f" % (i + trained_times, train_accuracy, loss))
        if i % 500 == 0 and i != 0:
            save_path = lstm_net.saver.save(sess, FLAGS.model_path + '/points-' + str(i + trained_times) + '.ckpt')
            logger.info("Model saved in file: {}".format(save_path))
            # print('loss: %.4f' % cross_entropy)
        # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        # saver.save(sess, FLAGS.model_path, global_step=i)
        sess.run([lstm_net.train_step],
                 feed_dict={lstm_net._X: batch[0], lstm_net.y: batch[1], lstm_net.keep_prob: FLAGS.train_keep_prob, lstm_net.batch_size: _batch_size})


def test(lstm_net, sess, ckpt):
    logger = myrnn2_utils.get_logger(FLAGS.test_log_file, logging.DEBUG)
    mnist = myrnn2_utils.load_data(FLAGS.data_dir, logger)
    load_model(sess, ckpt, lstm_net, logger)
    logger.info('accuracy: %.4f'%sess.run(lstm_net.accuracy, feed_dict={
        lstm_net._X: mnist.test.images, lstm_net.y: mnist.test.labels, lstm_net.keep_prob: FLAGS.test_keep_prob,
        lstm_net.batch_size: mnist.test.images.shape[0]}))


def predictOne(lstm_net, sess, ckpt):
    logger = myrnn2_utils.get_logger(FLAGS.predictOne_log_file, logging.DEBUG)
    mnist = myrnn2_utils.load_data(FLAGS.data_dir, logger)
    load_model(sess, ckpt, lstm_net, logger)
    X3 = mnist.train.images[10]
    img3 = X3.reshape([28, 28])
    # plt.imshow(img3, 'gray')
    # plt.show()
    X3.shape = [-1, 784]
    y_batch = mnist.train.labels[10]
    y_batch.shape = [-1, lstm_net.class_num]
    # X3_outputs.shape=(28, 1, 256)
    X3_outputs = np.array(sess.run(lstm_net.outputs, feed_dict={
        lstm_net._X: X3, lstm_net.y: y_batch, lstm_net.keep_prob: 0.5, lstm_net.batch_size: 1}))
    logger.info('X3_outputs: {}'.format(X3_outputs.shape))
    # X3_outputs.shape=(28, 256)
    np.reshape(X3_outputs, [-1, lstm_net.hidden_size])
    h_W, h_bias = sess.run([lstm_net.W, lstm_net.b], feed_dict={
        lstm_net._X: X3, lstm_net.y: y_batch, lstm_net.keep_prob: 1, lstm_net.batch_size: 1})
    h_bias = np.reshape(h_bias, [-1, 10])

    bar_index = range(lstm_net.class_num)
    for i in range(X3_outputs.shape[0]):
        plt.subplot(7, 4, i + 1)
        # X3_h_state.shape=(1,256)
        X3_h_shate = X3_outputs[i, :].reshape([-1, lstm_net.hidden_size])
        pro = sess.run(tf.nn.softmax(tf.matmul(X3_h_shate, h_W) + h_bias))
        plt.bar(bar_index, pro[0], width=0.2, align='center')
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    myrnn2_utils.make_folder()
    lstm_net = myrnn2_model.LSTM_net()
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        # test(lstm_net, sess, ckpt)
        # predictOne(lstm_net, sess, ckpt)
        train(lstm_net, sess, ckpt)

# tensorboard --logdir graph
# open you google chrome, type the link shown on your terminal or CMD. (something like this: http://localhost:6006)
