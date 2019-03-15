import tensorflow as tf
import matplotlib.pyplot as plt
import keras_metrics as km
import numpy as np

from datasets import Dataset
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
        tf.keras.layers.Dense(output_size, activation='sigmoid'),
    ])
    return model


model = build_model(OUTPUT_SIZE, 20)
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['binary_accuracy', km.binary_precision(), km.binary_recall(), km.binary_f1_score()])

inp = tf.placeholder(dtype=tf.float32, shape=(None, TIMESTEP, INPUT_SIZE))
output = model(inp)

train_dataset = Dataset("./dataset/" + house + "/LSTM_input_train_req.npy", batch_size=BATCH_SIZE)
test_dataset = Dataset("./dataset/" + house + "/LSTM_input_test_req.npy", batch_size=1, should_shuffle=False)

train_losses = []

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    for i in range(1000):
        x, y = train_dataset.next_batch()
        history = model.fit(x, y, batch_size=BATCH_SIZE)
        train_losses.append(history.history['loss'][0])

    # plt.plot(train_losses)
    # plt.show()

    print('Testing:')
    print('======================================')
    TEST_DAYS_NUM = 100
    THRESHOLD = 0.5

    for i in range(TEST_DAYS_NUM):
        # read day i
        print('DAY {} =================================================='.format(i))
        day_i_x, day_i_y = test_dataset.next_batch()
        day_i_output = model.predict(day_i_x)
        day_i_output[day_i_output < THRESHOLD] = 0
        day_i_output[day_i_output >= THRESHOLD] = 1

        scores = model.evaluate(day_i_x, day_i_y, verbose=0)
        print(scores)
        # distance between day_i output and day_i_y
        print(np.sum((day_i_output - day_i_y) ** 2))

        # day_i_output[0, ...], day_i_y[0, ...]
        pass
