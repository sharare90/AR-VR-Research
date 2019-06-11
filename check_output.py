import numpy as np
import tensorflow as tf
from datasets import Dataset
from setting import house, num_test_days, num_train_days, num_valid_requests, saved_model_folder, dict_reqs_nums
from LSTM import build_lstm_model, perform_testing, precision_threshold, recall_threshold, f1_score_threshold, \
    THRESHOLD, learning_rate

#   test with loading

model = build_lstm_model(num_valid_requests, 64)

model.compile(
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
    loss='binary_crossentropy',
    # metrics=['binary_accuracy', precision_threshold(0.3), km.binary_precision(), km.binary_recall(), km.binary_f1_score()]
    metrics=[tf.keras.metrics.binary_accuracy, precision_threshold(THRESHOLD), recall_threshold(THRESHOLD),
             f1_score_threshold(THRESHOLD)])
test_dataset = Dataset("./dataset/" + house + "/LSTM_input_validation_req.npy", batch_size=5, should_shuffle=False,
                       num_days_take=-1)

# model.predict(np.zeros([1, 24, num_valid_requests]))

false_neg = {}
false_pos = {}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.load_weights('./' + saved_model_folder + '/best_model')
    # for _ in range(num_train_days):
    # a, b = sess.run(test_dataset.next_item)
    # for r, i in enumerate(b[0, :, :]):
    #     for c, j in enumerate(i):
    #         if j == 1.0 and np.round(model.predict(a))[0, r, c] == 0.0:
    #             if c in false_neg:
    #                 false_neg[c] += 1
    #             else:
    #                 false_neg[c] = 1
    #         if j == 0.0 and np.round(model.predict(a))[0, r, c] == 1.0:
    #             if c in false_pos:
    #                 false_pos[c] += 1
    #             else:
    #                 false_pos[c] = 1
    # print(a)
    # print('prediction')
    # print(np.round(model.predict(a)))
    # print('output')
    # print(b)
    # print(false_neg)
    # print(false_pos)

    perform_testing(model, test_dataset.iterator, verbose=True)
