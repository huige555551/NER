# -*- coding=utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import warnings
import numpy as np
import logging
from myrnn2_config import FLAGS
import matplotlib.pyplot as plt
import myrnn2_utils

import functools

def define_scope(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class LSTM_net():
    def __init__(self):
        self.lr = FLAGS.lr
        self.training_steps = FLAGS.training_steps

        # netword parameter
        self.input_size = FLAGS.input_size
        self.timestep_size = FLAGS.timestep_size
        self.hidden_size = FLAGS.hidden_size
        self.layer_num = FLAGS.layer_num
        self.class_num = FLAGS.class_num

        with tf.variable_scope('Inputs'):
            self._X = tf.placeholder(tf.float32, [None, 784], name="_X")
            self.y = tf.placeholder(tf.float32, [None, self.class_num], name="y")
        # 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
        self.batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32, batch_size = 128
        self.keep_prob = tf.placeholder(tf.float32, [])
        # input [None, timesteps, num_input]
        self.X = tf.reshape(self._X, [-1, 28, 28])
        self.outputs = list()
        self._y_pre = None
        self._cross_entropy = None
        self._train_step = None
        self._correct_prediction = None
        self._accuracy = None
        self.y_pre
        self.cross_entropy
        self.train_step
        self.correct_prediction
        self.accuracy
        self.saver = tf.train.Saver()

    @property
    def y_pre(self):
        if self._y_pre is None:
            self.h_state = self.lstm_net()
            # self.h_state = self.lstm()
            self.W = tf.Variable(tf.random_normal([self.hidden_size, self.class_num], 0, 0.1, tf.float32))
            self.b = tf.Variable(tf.random_normal([self.class_num], 0, 0.1, tf.float32))
            self._y_pre = tf.nn.softmax(tf.matmul(self.h_state, self.W) + self.b)
        return self._y_pre

    @property
    def cross_entropy(self):
        if self._cross_entropy is None:
            self._cross_entropy = tf.reduce_mean(
                -tf.reduce_sum(self.y * tf.log(self.y_pre), reduction_indices=[1]))  # 损失函数，交叉熵
        return self._cross_entropy

    @property
    def train_step(self):
        if self._train_step is None:
            self._train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)  # 使用adam优化
        return self._train_step

    @property
    def correct_prediction(self):
        if self._correct_prediction is None:
            self._correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_pre, 1))  # 计算准确度
        return self._correct_prediction

    @property
    def accuracy(self):
        if self._accuracy is None:
            self._accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        return self._accuracy

    # lstm_cell和lstm_net可以简化为下面这个函数
    def lstm(self):
        x = tf.unstack(self.X, self.timestep_size, 1)
        lstm_cell = rnn.BasicLSTMCell(self.hidden_size)
        (outputs, states) = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        return outputs[-1]

    def lstm_cell(self):
        lstm_cell = rnn.BasicLSTMCell(self.hidden_size)
        lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
        return lstm_cell

    def lstm_net(self):
        mlstm_cell = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)])
        init_state = mlstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        state = init_state
        with tf.variable_scope('RNN'):
            for timestep in range(self.timestep_size):
                (cell_output, state) = mlstm_cell(self.X[:, timestep, :], state)
                self.outputs.append(cell_output)
        # outputs.shape = [batch_size, timestep_size, hidden_size]
        h_state = self.outputs[-1]
        return h_state


    # def _build_net(self):
        # self.h_state = self.lstm_net()
        # # self.h_state = self.lstm()
        # self.W = tf.Variable(tf.random_normal([self.hidden_size, self.class_num], 0, 0.1, tf.float32))
        # self.b = tf.Variable(tf.random_normal([self.class_num], 0, 0.1, tf.float32))
        # y_pre = tf.nn.softmax(tf.matmul(self.h_state, self.W) + self.b)

        # Define loss and optimizer
        # self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.y_pre), reduction_indices=[1]))  # 损失函数，交叉熵
        # self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)  # 使用adam优化

        # Evaluate model (with test logits, for dropout to be disabled)
        # self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_pre, 1))  # 计算准确度
        # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # self.saver = tf.train.Saver()