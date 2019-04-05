import tensorflow as tf
import numpy as np


# from LSTM import BATCH_SIZE, TIMESTEP, INPUT_SIZE


# BATCH_SIZE, TIMESTEP, INPUT_SIZE
from setting import num_actions


class Dataset(object):
    def __init__(self, dataset_address, batch_size, should_shuffle=True):
        def get_x_y(example):
            example = tf.cast(example, tf.float32)
            return tf.concat((tf.zeros(shape=(1, num_actions)), example[:23, :]), axis=0), example

        self.address = dataset_address
        self.data = np.load(self.address)
        self.batch_size = batch_size
        self.tf_dataset = tf.data.Dataset.from_tensor_slices(self.data)

        self.tf_dataset = self.tf_dataset.batch(24)

        if should_shuffle:
            self.tf_dataset = self.tf_dataset.shuffle(buffer_size=1000)

        self.tf_dataset = self.tf_dataset.map(get_x_y)
        self.tf_dataset = self.tf_dataset.repeat(-1).batch(self.batch_size)

        self.iterator = self.tf_dataset.make_one_shot_iterator()
        self.next_item = self.iterator.get_next()


if __name__ == '__main__':
    dataset = Dataset("./dataset/house2/LSTM_input_train_req.npy", batch_size=8)
    with tf.Session() as sess:
        for _ in range(10):
            x, y = sess.run(dataset.next_item)
            print(x)
            print(y)
