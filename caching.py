from setting import house, phase, caching_cost_each_hour, no_respond, num_valid_requests, dict_reqs_nums, \
    number_of_caching_trial, saved_model_folder, threshold_prob_based, delay_d, size, relative_quality, delay, max_score
import numpy as np
from keras.models import load_model
import tensorflow as tf


class Caching(object):
    def __init__(self):
        self.data_file = "./dataset_copy/" + house + "/LSTM_input_" + 'test' + "_req.npy"
        self.data = np.load(self.data_file)
        self.caching_cost_each_hour = caching_cost_each_hour
        self.no_respond = no_respond
        print(np.sum(self.data) * self.caching_cost_each_hour)

    def random_caching(self):
        average_cost = 0
        average_score = 0

        for i in range(number_of_caching_trial):
            cost = 0
            score = 0
            for line in self.data:
                line = line.astype(np.int32)
                for j in range(num_valid_requests):
                    random_action_num = np.random.randint(0, 2)
                    if random_action_num == 1:
                        cost += self.caching_cost_each_hour
                        if line[j] == 1:
                            score += relative_quality * max_score
                    elif line[j] == 1 and random_action_num == 0:
                        cost += self.no_respond
                        score += (delay_d ** delay) * relative_quality * max_score
            average_cost += cost
            average_score += score
        average_cost /= number_of_caching_trial
        average_score /= number_of_caching_trial
        return average_cost, average_score

    def cache_everything(self):
        cost = 0
        score = 0
        for line in self.data:
            for j in range(num_valid_requests):
                cost += self.caching_cost_each_hour
                if line[j] == 1:
                    score += relative_quality * max_score
        return cost, score

    def average_based_caching(self, threshold=0):
        processed_data_file = "./dataset_copy/" + house + "/processed_data_train.txt"
        cache_data = np.zeros(shape=(24, num_valid_requests))

        with open(processed_data_file) as data:
            line_counter = 0
            for line in data:
                line = line[1:-2]
                for info in line.split(','):
                    action_name = ''
                    repeat_num = 0
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
        score = 0
        interval_count = 0
        for line in self.data:
            line = line.astype(np.int32)
            for j in range(num_valid_requests):
                should_cache = cache_data[interval_count, j]

                if should_cache == 1:
                    cost += self.caching_cost_each_hour
                    if line[j] == 1:
                        score += relative_quality * max_score
                elif line[j] == 1 and should_cache == 0:
                    cost += self.no_respond
                    score += (delay_d ** delay) * relative_quality * max_score

            interval_count += 1
            interval_count = interval_count % 24
        return cost, score

    def LSTM_based_caching(self, threshold=0.5):
        from LSTM import build_lstm_model, OUTPUT_SIZE
        model = build_lstm_model(OUTPUT_SIZE, 32)
        cost = 0
        score = 0

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            model.load_weights('./saved_models_for_plot/' + saved_model_folder + '_1h/best_model')
            # model.load_weights('./' + saved_model_folder + '/best_model')

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
                for i in range(0, 24):
                    for j in range(num_valid_requests):
                        if prediction_current_day[i, j] == 1:
                            cost += self.caching_cost_each_hour
                            if current_day_data[i, j] == 1:
                                score += relative_quality * max_score
                        elif current_day_data[i, j] == 1 and prediction_current_day[i, j] == 0:
                            cost += self.no_respond
                            score += (delay_d ** delay) * relative_quality * max_score
        return cost, score

    def majority_vote_caching(self, threshold=0.5):
        from LSTM import build_lstm_model, OUTPUT_SIZE
        model1 = build_lstm_model(OUTPUT_SIZE, 32)
        model2 = build_lstm_model(OUTPUT_SIZE, 32)
        model3 = build_lstm_model(OUTPUT_SIZE, 32)
        model4 = build_lstm_model(OUTPUT_SIZE, 32)
        model5 = build_lstm_model(OUTPUT_SIZE, 32)
        model6 = build_lstm_model(OUTPUT_SIZE, 32)
        model7 = build_lstm_model(OUTPUT_SIZE, 32)
        model8 = build_lstm_model(OUTPUT_SIZE, 32)
        model9 = build_lstm_model(OUTPUT_SIZE, 32)
        model10 = build_lstm_model(OUTPUT_SIZE, 32)
        model11 = build_lstm_model(OUTPUT_SIZE, 32)
        model12 = build_lstm_model(OUTPUT_SIZE, 32)
        model13 = build_lstm_model(OUTPUT_SIZE, 32)
        model14 = build_lstm_model(OUTPUT_SIZE, 32)
        model15 = build_lstm_model(OUTPUT_SIZE, 32)

        cost = 0
        score = 0

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            if house == 'house1':
                model1.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d1_dropout0/best_model')
                model2.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d1_dropout2_epochs300/best_model')
                model3.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d1_dropout8_epochs500/best_model')
                model4.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d1_lr0001_epochs1000/best_model')
                model5.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d1_threshold005_Epochs1000/best_model')
                model6.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d1_withbatchnormalization_epochs225/best_model')
                model7.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d1_withl1l2regu_dropout5_epochs225/best_model')
                model8.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d1_withl1l2regu_dropout5_epochs300/best_model')
                model9.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d1_withl1regu_dropout5_epochs225/best_model')
                model10.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d1_withl1regu_dropout5_epochs300/best_model')
                model11.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d1_withl2regu_dropout0_epochs200/best_model')
                model12.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d1_withl2regu_dropout0_lr0001_epochs1000/best_model')
                model13.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d1_withl2regu_dropout5_epochs225/best_model')
                model14.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d1_withl2regu_dropout5_epochs300/best_model')
                model15.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d1_withoutl1regu_dropout5_epochs300/best_model')
            # model.load_weights('./' + saved_model_folder + '/best_model')
            elif house == 'house2':
                model1.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d2_withoutregu_dropout2_epochs225/best_model')
                model2.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d2_withoutregu_dropout5_epochs225/best_model')
                model3.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d2_withoutregu_dropout5_lr0001_epochs1000/best_model')
                model4.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d2_withoutregu_dropout8_epochs225/best_model')
                model5.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d2_withoutregu_dropout8_epochs500/best_model')
                model6.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d2_withregul101_dropout0_epochs225/best_model')
                model7.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d2_withregul202_dropout0_epochs225/best_model')
                model8.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d2_withbatchnormalization_epochs225/best_model')
                model9.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d2_withl2regu_dropout5_epochs300/best_model')
                model10.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d2_withl2regu_dropout0_lr0001_epochs1000/best_model')
                model11.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d2_threshold005_Epochs1000/best_model')
                model12.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d2_withl2regu_dropout0_lr0001_epochs1000/best_model')
                model13.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d2_dropout0/best_model')
                model14.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d2_withl1l2regu_dropout5_epochs225/best_model')
                model15.load_weights(
                    './models_for_majority_voting/' + saved_model_folder + '/saved_models_d2_withl1l2regu_dropout5_epochs300/best_model')

            current_day = 0
            predictions = []
            while True:
                current_day_data = self.data[24 * current_day: 24 * current_day + 24, :]
                current_day += 1
                if current_day_data.shape[0] != 24:
                    break

                predictions.append(model1.predict(current_day_data.reshape((1, 24, num_valid_requests))))
                predictions.append(model2.predict(current_day_data.reshape((1, 24, num_valid_requests))))
                predictions.append(model3.predict(current_day_data.reshape((1, 24, num_valid_requests))))
                predictions.append(model4.predict(current_day_data.reshape((1, 24, num_valid_requests))))
                predictions.append(model5.predict(current_day_data.reshape((1, 24, num_valid_requests))))
                predictions.append(model6.predict(current_day_data.reshape((1, 24, num_valid_requests))))
                predictions.append(model7.predict(current_day_data.reshape((1, 24, num_valid_requests))))
                predictions.append(model8.predict(current_day_data.reshape((1, 24, num_valid_requests))))
                predictions.append(model9.predict(current_day_data.reshape((1, 24, num_valid_requests))))
                predictions.append(model10.predict(current_day_data.reshape((1, 24, num_valid_requests))))
                predictions.append(model11.predict(current_day_data.reshape((1, 24, num_valid_requests))))
                predictions.append(model12.predict(current_day_data.reshape((1, 24, num_valid_requests))))
                predictions.append(model13.predict(current_day_data.reshape((1, 24, num_valid_requests))))
                predictions.append(model14.predict(current_day_data.reshape((1, 24, num_valid_requests))))
                predictions.append(model15.predict(current_day_data.reshape((1, 24, num_valid_requests))))

                current_day_data = current_day_data.astype(np.int32)

                for pred in predictions:
                    pred[pred < threshold] = 0
                    pred[pred >= threshold] = 1

                # prediction_current_day = current_day_data.reshape((24, 13))[:, :].astype(np.int32)
                prediction_current_day = np.zeros((24, num_valid_requests))
                for m in range(0, 24):
                    for n in range(num_valid_requests):
                        num_ones = 0
                        num_zeros = 0
                        for pred in predictions:
                            pred = pred[0].astype(np.int32)
                            if pred[m, n] == 1:
                                num_ones += 1
                            elif pred[m, n] == 0:
                                num_zeros += 1
                        if num_ones >= num_zeros:
                            prediction_current_day[m, n] = 1
                        else:
                            prediction_current_day[m, n] = 0

                for i in range(0, 24):
                    for j in range(num_valid_requests):
                        if prediction_current_day[i, j] == 1:
                            cost += self.caching_cost_each_hour
                            if current_day_data[i, j] == 1:
                                score += relative_quality * max_score
                        elif current_day_data[i, j] == 1 and prediction_current_day[i, j] == 0:
                            cost += self.no_respond
                            score += (delay_d ** delay) * relative_quality * max_score
        return cost, score


cache = Caching()
random_cost, random_score = cache.random_caching()
cache_everything_cost, cache_everything_score = cache.cache_everything()
average_based_cost, average_based_score = cache.average_based_caching(threshold_prob_based)
LSTM_based_cost, LSTM_based_score = cache.LSTM_based_caching(0.75)
majority_voting_cost, majority_voting_score = cache.majority_vote_caching(0.75)
print(random_cost, random_score)
print(cache_everything_cost, cache_everything_score)
print(average_based_cost, average_based_score)
print(LSTM_based_cost, LSTM_based_score)
print(majority_voting_cost, majority_voting_score)
#
# cache.caching_cost_each_hour = 0
# cache.no_respond = 0
# result = []
# for i in range(3):
#     for j in range(1):
#         result.append([cache.caching_cost_each_hour, cache.no_respond, cache.average_based_caching(15)])
#         cache.caching_cost_each_hour += 1
#     cache.no_respond += 1
# np.save("./saved_models_d1/caching_Average_costs.npy", result)
# np.savetxt("./saved_models_d1/caching_Average_costs.txt", result)

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
