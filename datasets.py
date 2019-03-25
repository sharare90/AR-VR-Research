import tensorflow as tf
import numpy as np
# from LSTM import BATCH_SIZE, TIMESTEP, INPUT_SIZE


# BATCH_SIZE, TIMESTEP, INPUT_SIZE


class Dataset(object):
    def __init__(self, dataset_address, batch_size, should_shuffle=True):

        self.address = dataset_address
        self.data = np.load(self.address)
        self.batch_size = batch_size
        self.tf_dataset = tf.data.Dataset.from_tensor_slices(self.data)

        self.tf_dataset = self.tf_dataset.batch(24)

        if should_shuffle:
            self.tf_dataset = self.tf_dataset.shuffle(buffer_size=1000)

        self.tf_dataset = self.tf_dataset.repeat(-1).batch(self.batch_size)
        self.iterator = self.tf_dataset.make_one_shot_iterator()
        self.next_item = self.iterator.get_next()
        self.sess = tf.Session()

    def next_batch(self):
        next_item_np = self.sess.run(self.next_item)

        x = np.zeros(next_item_np.shape, dtype=np.float32)
        x[:, 1:, :] = next_item_np[:, :23, :]

        y = next_item_np
        return x, y

