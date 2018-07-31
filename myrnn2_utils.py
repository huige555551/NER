from tensorflow.examples.tutorials.mnist import input_data
import logging
import os

def load_data(data_dir, logger):
    mnist = input_data.read_data_sets(data_dir, one_hot=True)
    logger.info('data_dir: ' + data_dir)
    logger.info('test image shape: {}' .format(mnist.test.images.shape))
    logger.info('test labels shape: {}'.format(mnist.test.labels.shape))
    logger.info('train image shape: {}'.format(mnist.train.images.shape))
    logger.info('train labels shape: {}'.format(mnist.train.labels.shape))
    return mnist


def get_logger(log_file, level):
    logger = logging.getLogger(log_file)
    logger.setLevel(level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

def make_folder():
    if not os.path.isdir("log"):
        os.makedirs("log")
    if not os.path.isdir("model"):
        os.makedirs("model")
