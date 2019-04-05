import tensorflow as tf
import numpy as np
import tensorboard
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from datasets import Dataset
from setting import house, phase, num_valid_requests, num_train_days

INPUT_SIZE = num_valid_requests
OUTPUT_SIZE = num_valid_requests
TIMESTEP = 24
BATCH_SIZE = 6
THRESHOLD = 0.5
NUM_EPOCHS = 2000


def run():
    model = build_model(OUTPUT_SIZE, 64)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        # metrics=['binary_accuracy', precision_threshold(0.3), km.binary_precision(), km.binary_recall(), km.binary_f1_score()]
        metrics=[precision_threshold(THRESHOLD), recall_threshold(THRESHOLD), f1_score_threshold(THRESHOLD)]
    )

    inp = tf.placeholder(dtype=tf.float32, shape=(None, TIMESTEP, INPUT_SIZE))
    output = model(inp)

    train_dataset = Dataset("./dataset/" + house + "/LSTM_input_train_req.npy", batch_size=BATCH_SIZE)
    validation_dataset = Dataset("./dataset/" + house + "/LSTM_input_validation_req.npy", batch_size=5,
                                 should_shuffle=False)
    test_dataset = Dataset("./dataset/" + house + "/LSTM_input_test_req.npy", batch_size=1, should_shuffle=False)

    train_losses = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        callbacks = [
            # EarlyStopping(monitor='val_f1_score', patience=50, mode='max', baseline=0.40),
            ModelCheckpoint(
                filepath='./saved_models_d1/best_model',
                monitor='val_f1_score',
                save_best_only=True,
                save_weights_only=True,
                mode='max'
            )
        ]

        history = model.fit(
            train_dataset.iterator,
            steps_per_epoch=int(num_train_days / BATCH_SIZE) + 1,
            epochs=NUM_EPOCHS,
            verbose=True,
            validation_data=validation_dataset.iterator,
            validation_steps=1,
            callbacks=callbacks
        )
        np.savetxt('./saved_models_d1/train_loss.txt', history.history['loss'])
        np.savetxt('./saved_models_d1/val_loss.txt', history.history['val_loss'])
        np.savetxt('./saved_models_d1/train_f1score.txt', history.history['f1_score'])
        np.savetxt('./saved_models_d1/val_f1score.txt', history.history['val_f1_score'])
        plt.plot(history.history['f1_score'])
        plt.plot(history.history['val_f1_score'])
        plt.show()

        # for i in range(20000):
        #     x, y = train_dataset.next_batch()
        #     verbose = i % 1000 == 0
        #     if verbose:
        #         print('iteration {}'.format(i))
        #
        #     history = model.fit(x, y, batch_size=BATCH_SIZE, verbose=verbose, epochs=1,
        #                         validation_split=0.2)
        #     if verbose:
        #         perform_testing(model, test_dataset)
        #         plt.plot(history.history['loss'])
        #
        #     train_losses.append(history.history['loss'][0])

        # plt.plot(train_losses)
        # plt.show()
        model.save_weights('./saved_models_d1/LSTM_model')
        perform_testing(model, test_dataset.iterator, verbose=True, write=True)


def perform_testing(model, test_dataset, verbose=False, write=False):
    print('Testing:')
    print('======================================')
    TEST_DAYS_NUM = 10

    # mean_precision = 0
    # mean_recall = 0
    # mean_f1_score = 0

    model.evaluate(test_dataset, verbose=True, steps=2)
    # with open('./dataset/' + house + '/predicted_' + phase + '.csv', 'w') as predicted:
    #     for i in range(TEST_DAYS_NUM):
    #         # read day i
    #         if verbose:
    #             print('DAY {} =================================================='.format(i))
    #         day_i_x, day_i_y = test_dataset.
    #         day_i_output = model.predict(day_i_x)
    #         day_i_output[day_i_output < THRESHOLD] = 0
    #         day_i_output[day_i_output >= THRESHOLD] = 1
    #         # print(day_i_output)
    #
    #         scores = model.evaluate(day_i_x, day_i_y, verbose=0)
    #         mean_precision += scores[1]
    #         mean_recall += scores[2]
    #         mean_f1_score += scores[3]
    #         if verbose:
    #             print(scores)
    #             # distance between day_i output and day_i_y
    #             print(np.sum((day_i_output - day_i_y)))
    #
    #             # day_i_output[0, ...], day_i_y[0, ...]

    # mean_precision /= TEST_DAYS_NUM
    # mean_recall /= TEST_DAYS_NUM
    # mean_f1_score /= TEST_DAYS_NUM

    # if verbose:
    #     print('mean precision: {}'.format(mean_precision))
    #     print('mean recall: {}'.format(mean_recall))
    # print('mean f1 score: {}'.format(mean_f1_score))


def build_model(output_size, rnn_units):
    rnn = tf.keras.layers.LSTM
    model = tf.keras.Sequential([

        rnn(32, activation='relu',
            return_sequences=True,
            unroll=True,
            recurrent_initializer='orthogonal',
            stateful=False),
        #
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(rate=0.5),
        # tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        # tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(output_size, activation='sigmoid'),
    ])
    return model


def precision_threshold(threshold=0.5):
    def precision(y_true, y_predict):
        threshold_value = threshold
        y_predict = tf.cast(tf.greater(tf.clip_by_value(y_predict, 0, 1), threshold_value), tf.float32)
        true_positives = tf.round(tf.reduce_sum(tf.clip_by_value(y_true * y_predict, 0, 1)))
        predicted_positives = tf.reduce_sum(y_predict)
        precision_ratio = true_positives / (predicted_positives + 10e-6)

        return precision_ratio

    return precision


def recall_threshold(threshold=0.5):
    def recall(y_true, y_predict):
        threshold_value = threshold
        y_predict = tf.cast(tf.greater(tf.clip_by_value(y_predict, 0, 1), threshold_value), tf.float32)
        true_positives = tf.round(tf.reduce_sum(tf.clip_by_value(y_true * y_predict, 0, 1)))
        possible_positives = tf.reduce_sum(tf.clip_by_value(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + 10e-6)

        return recall_ratio

    return recall


def f1_score_threshold(threshold=0.5):
    def f1_score(y_true, y_predict):
        threshold_value = threshold
        y_predict = tf.cast(tf.greater(tf.clip_by_value(y_predict, 0, 1), threshold_value), tf.float32)
        true_positives = tf.round(tf.reduce_sum(tf.clip_by_value(y_true * y_predict, 0, 1)))
        predicted_positives = tf.reduce_sum(y_predict)
        precision_ratio = true_positives / (predicted_positives + 10e-6)

        possible_positives = tf.reduce_sum(tf.clip_by_value(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + 10e-6)

        return (2 * recall_ratio * precision_ratio) / (recall_ratio + precision_ratio + 10e-6)

    return f1_score


#
if __name__ == '__main__':
    #   test with loading
    # test_dataset = Dataset("./dataset/" + house + "/LSTM_input_test_req.npy", batch_size=1, should_shuffle=False)
    #
    # model = build_model(OUTPUT_SIZE, 64)
    # model.compile(
    #     optimizer='adam',
    #     loss='binary_crossentropy',
    #     metrics=[precision_threshold(THRESHOLD), recall_threshold(THRESHOLD), f1_score_threshold(THRESHOLD)]
    # )
    # model.predict(np.zeros([1, 24, 8]))
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     model.load_weights('./saved_models_d1/best_model')
    #
    #     for _ in range(2):
    #         a, b = sess.run(test_dataset.next_item)
    #         print(a)
    #         print('prediction')
    #         print(np.round(model.predict(a)))
    #         print('output')
    #         print(b)
    #
    #     perform_testing(model, test_dataset.iterator, verbose=True)

    #  run training
    run()
