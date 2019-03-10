import numpy as np
from setting import len_intervals, house, num_valid_requests, dict_reqs_nums

file_name = house

each_day_info = np.zeros(shape=(int(24 / len_intervals), num_valid_requests))
previous_interval = 0
with open("./dataset/" + file_name + "/LSTM_input_train_req.txt", "w") as processed_data:
    with open("./dataset/house2_acts_reqs_train.csv") as input_file:

        for line in input_file:
            line = line.split(",")
            interval = int(line[1])
            if interval >= previous_interval:
                previous_interval = interval
                req = dict_reqs_nums[line[3][:-1]]
                each_day_info[interval][req] = 1
            else:
                previous_interval = 0
                req = dict_reqs_nums[line[3][:-1]]
                for i in each_day_info:
                    processed_data.write(str(i) + "\n")

                each_day_info = np.zeros(shape=(int(24 / len_intervals), num_valid_requests))
                each_day_info[interval][req] = 1
        for i in each_day_info:
            processed_data.write(str(i) + "\n")
