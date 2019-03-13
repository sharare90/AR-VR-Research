import numpy as np
from setting import len_intervals, house, num_valid_requests, dict_reqs_nums, phase

file_name = house

each_day_info = np.zeros(shape=(int(24 / len_intervals), num_valid_requests))
all_days = []
previous_interval = 0
with open("./dataset/" + file_name + "/LSTM_input_" + phase + "_req.txt", "w") as processed_data:
    with open("./dataset/" + house + "_acts_reqs_" + phase + ".csv") as input_file:

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
                all_days.append(each_day_info)
                # for i in each_day_info:
                #     processed_data.write(str(i)[1:-1] + "\n")

                each_day_info = np.zeros(shape=(int(24 / len_intervals), num_valid_requests))
                each_day_info[interval][req] = 1
        # for i in each_day_info:
        #     processed_data.write(str(i)[1:-1] + "\n")
        all_days.append(each_day_info)
        numpy_array = np.concatenate(all_days, axis=0)
        np.save("./dataset/" + file_name + "/LSTM_input_" + phase + "_req", numpy_array)

