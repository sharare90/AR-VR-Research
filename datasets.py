import tensorflow as tf
import numpy as np


# BATCH_SIZE, TIMESTEP, INPUT_SIZE


class DatasetTrain(object):
    def __init__(self, dataset_address, batch_size):
        self.address = dataset_address
        self.data = np.load(self.address)
        self.batch_size = batch_size
        self.tf_dataset = tf.data.Dataset.from_tensor_slices(self.data)
        self.tf_dataset = self.tf_dataset.batch(24).shuffle(buffer_size=1000).batch(self.batch_size).repeat(-1)
        self.iterator = self.tf_dataset.make_one_shot_iterator()
        self.next_item = self.iterator.get_next()

    def next_batch(self):
        with tf.Session() as sess:
            next_item_np = sess.run(self.next_item)

            x = np.zeros(next_item_np.shape, dtype=np.float32)
            x[:, 1:, :] = next_item_np[:, :23, :]

            y = next_item_np
            return x, y


class DatasetTest(object):
    def __init__(self, dataset_address, batch_size):
        self.address = dataset_address
        self.data = np.load(self.address)
        self.batch_size = batch_size
        self.tf_dataset = tf.data.Dataset.from_tensor_slices(self.data)
        self.tf_dataset = self.tf_dataset.batch(24).batch(self.batch_size).repeat(-1)
        self.iterator = self.tf_dataset.make_one_shot_iterator()
        self.next_item = self.iterator.get_next()

    def next_batch(self):
        with tf.Session() as sess:
            next_item_np = sess.run(self.next_item)

            x = np.zeros(next_item_np.shape, dtype=np.float32)
            x[:, 1:, :] = next_item_np[:, :23, :]

            y = next_item_np
            return x, y
