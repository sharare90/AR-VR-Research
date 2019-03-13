import tensorflow as tf
import numpy as np

from datasets import DatasetTest, DatasetTrain
from setting import house

INPUT_SIZE = 5
OUTPUT_SIZE = 5
TIMESTEP = 24
BATCH_SIZE = 2


def build_model(output_size, rnn_units):
    rnn = tf.keras.layers.LSTM
    model = tf.keras.Sequential([

        rnn(rnn_units,
            return_sequences=True,
            recurrent_initializer='glorot_uniform',
            stateful=False),
        tf.keras.layers.Dense(output_size)
    ])
    return model


model = build_model(OUTPUT_SIZE, 20)
model.compile(optimizer='adam', loss='mean_squared_error')

inp = tf.placeholder(dtype=tf.float32, shape=(None, TIMESTEP, INPUT_SIZE))
output = model(inp)

train_dataset = DatasetTrain("./dataset/" + house + "/LSTM_input_train_req.npy", batch_size=BATCH_SIZE)
test_dataset = DatasetTest("./dataset/" + house + "/LSTM_input_test_req.npy", batch_size=1)
for i in range(1):
    x, y = train_dataset.next_batch()
    model.fit(x, y, batch_size=BATCH_SIZE)


print('Testing:')
print('======================================')
TEST_DAYS_NUM = 100

for i in range(TEST_DAYS_NUM):
    # read day i
    day_i_x, day_i_y = test_dataset.next_batch()
    day_i_output = model.predict(day_i_x)

    # distance between day_i output and day_i_y
    print(day_i_y)
    print(day_i_output)
