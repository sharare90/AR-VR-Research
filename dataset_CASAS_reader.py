import csv

len_intervals = 2
file_name = "house1"
list_of_intervals = [i for i in range(0, 24, len_intervals)]
list_of_tasks_frequency = [{} for i in list_of_intervals]
list_of_daily_tasks = list()
today_list = [0 for i in range(0, 24, len_intervals)]
interval_task = []
previous_date = "2009-06-10"
with open("./dataset/" + file_name + "/processed_data.txt", "w") as processed_data:
    with open("./dataset/" + file_name + ".csv") as input_file:
        input_data = csv.reader(input_file, delimiter=",")

        for line in input_data:
            date = line[0]
            time = int(line[1].split(":")[0])
            task = line[4]
            i = int(time / len_intervals)
            if "begin" in task and file_name == 'house2':
                interval_task.append(str(i) + "," + str(task[:-6]))
            elif file_name == 'house1':
                interval_task.append(str(i) + "," + task)
            tasks_frequency_i = list_of_tasks_frequency[i]
            if task in tasks_frequency_i:
                tasks_frequency_i[task] += 1
            else:
                tasks_frequency_i[task] = 1

            if date != previous_date:
                list_of_daily_tasks.append(today_list)
                today_list = [0 for i in range(0, 24, len_intervals)]

            today_list[i] += 1
            previous_date = date

        list_of_daily_tasks.append(today_list)

    for i in list_of_tasks_frequency:
        processed_data.write(str(i) + "\n")

with open("./dataset/" + file_name + "/daily_tasks.txt", "w") as daily_tasks:
    for i in range(len(list_of_daily_tasks)):
        daily_tasks.write(str(list_of_daily_tasks[i]) + "\n")

with open("./dataset/" + file_name + "/time_task.csv", "w") as time_task:
    for i in interval_task:
        time_task.write(i + "\n")
