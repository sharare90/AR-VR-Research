from setting import house, phase, caching_cost_each_hour, no_respond, num_valid_requests, dict_reqs_nums, \
    number_of_caching_trial
import numpy as np
from keras.models import load_model
import tensorflow as tf


class Caching(object):
    def __init__(self):
        self.data_file = "./dataset/" + house + "/LSTM_input_" + 'test' + "_req.npy"
        self.data = np.load(self.data_file)
        print(np.sum(self.data))

    def random_caching(self):
        average_cost = 0

        for i in range(number_of_caching_trial):
            cost = 0
            for line in self.data:
                line = line.astype(np.int32)
                for j in range(num_valid_requests):
                    random_action_num = np.random.randint(0, 2)
                    if random_action_num == 1:
                        cost += caching_cost_each_hour
                    elif line[j] == 1 and random_action_num == 0:
                        cost += no_respond
            average_cost += cost
        average_cost /= number_of_caching_trial
        return average_cost

    def cache_everything(self):
        cost = 0
        for line in self.data:
            for j in range(num_valid_requests):
                cost += caching_cost_each_hour
        return cost

    def average_based_caching(self, threshold=0):
        processed_data_file = "./dataset/" + house + "/processed_data_train.txt"
        cache_data = np.zeros(shape=(24, num_valid_requests))

        with open(processed_data_file) as data:
            line_counter = 0
            for line in data:
                line = line[1:-2]
                for info in line.split(','):
                    info = info.strip()
                    if info:
                        action_name, repeat_num = info.split(':')
                    action_name = action_name[1:-1]
                    action_name = action_name.replace(' end', '').replace(' begin', '')
                    repeat_num = int(repeat_num)
                    if repeat_num > threshold:
                        cache_data[line_counter, dict_reqs_nums[action_name]] = 1

                line_counter += 1

        cost = 0
        interval_count = 0
        for line in self.data:
            line = line.astype(np.int32)
            for j in range(num_valid_requests):
                should_cache = cache_data[interval_count, j]

                if should_cache == 1:
                    cost += caching_cost_each_hour
                elif line[j] == 1 and should_cache == 0:
                    cost += no_respond

            interval_count += 1
            interval_count = interval_count % 24
        return cost

    def LSTM_based_caching(self, threshold=0.5):
        from LSTM import build_model, OUTPUT_SIZE
        model = build_model(OUTPUT_SIZE, 64)
        cost = 0

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            # model.load_weights('./saved_models_d1/LSTM_model')
            model.load_weights('./saved_models_d1/best_model')

            current_day = 0
            while True:
                current_day_data = self.data[24 * current_day: 24 * current_day + 24, :]
                current_day += 1
                if current_day_data.shape[0] != 24:
                    break

                prediction_current_day = model.predict(current_day_data.reshape((1, 24, num_valid_requests)))
                current_day_data = current_day_data.astype(np.int32)
                prediction_current_day[prediction_current_day < threshold] = 0
                prediction_current_day[prediction_current_day >= threshold] = 1
                prediction_current_day = prediction_current_day[0].astype(np.int32)
                # prediction_current_day = current_day_data.reshape((24, 13))[:, :].astype(np.int32)
                for i in range(2, 24):
                    for j in range(num_valid_requests):
                        if prediction_current_day[i, j] == 1:
                            cost += caching_cost_each_hour
                        elif current_day_data[i, j] == 1 and prediction_current_day[i, j] == 0:
                            cost += no_respond
        return cost


cache = Caching()
random_cost = cache.random_caching()
cache_everything_cost = cache.cache_everything()
cache_LSTM_based = cache.LSTM_based_caching(0.9)
cache_average_based = cache.average_based_caching(15)
print(random_cost)
print(cache_everything_cost)
print(cache_LSTM_based)
print(cache_average_based)



# min_threshold = 0
# min_value = 10000000
# for i in range(100):
#     cache_lstm_based = cache.LSTM_based_caching(i / 100)
#     print('\nThreshold:')
#     print(i / 100)
#     print(cache_lstm_based)
#     if cache_lstm_based < min_value:
#         min_threshold = i
#         min_value = cache_lstm_based
#
# print(min_threshold)


# min_threshold = 0
# min_value = 10000000
# for i in range(100):
#     cache_average_based = cache.average_based_caching(i)
#     if cache_average_based < min_value:
#         min_threshold = i
#         min_value = cache_average_based
# print(min_threshold)
# print(random_cost)