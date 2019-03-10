import tensorflow as tf
from setting import len_intervals, num_valid_requests

requests_in_one_interval = num_valid_requests  # number of valid requests
interval_steps = 24 / len_intervals  # number of intervals in a single day
num_classes = num_valid_requests
lstm_size = 50

# tf Graph input
X = tf.placeholder("float", [None, interval_steps, requests_in_one_interval])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {'out': tf.Variable(tf.random_normal([lstm_size, num_classes]))}
biases = {'out': tf.Variable(tf.random_normal([num_classes]))}
