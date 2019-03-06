import csv
from setting import len_intervals, house

file_name = house
list_of_intervals = [i for i in range(0, 24, len_intervals)]
list_of_tasks_frequency = [{} for i in list_of_intervals]
list_of_daily_tasks = list()
today_list = [0 for i in range(0, 24, len_intervals)]
interval_task = []
with open("./dataset/" + file_name + "/processed_data_train_req.txt", "w") as processed_data:
    with open("./dataset/house2_acts_reqs_train.csv") as input_file:

        for line in input_file:
            line = line.split(",")
            time_interval = int(line[1])
            task = line[3][]

            tasks_frequency_i = list_of_tasks_frequency[time_interval]
            if task in tasks_frequency_i:
                tasks_frequency_i[task] += 1
            else:
                tasks_frequency_i[task] = 1

            today_list[time_interval] += 1

        list_of_daily_tasks.append(today_list)

    for i in list_of_tasks_frequency:
        processed_data.write(str(i) + "\n")

with open("./dataset/" + file_name + "/daily_tasks_train_req.txt", "w") as daily_tasks:
    for i in range(len(list_of_daily_tasks)):
        daily_tasks.write(str(list_of_daily_tasks[i]) + "\n")

with open("./dataset/" + file_name + "/time_task_train_req.csv", "w") as time_task:
    for i in interval_task:
        time_task.write(i + "\n")
