import csv

len_intervals = 1
list_of_intervals = [i for i in range(0, 24, len_intervals)]
list_of_time_tasks = [{} for i in list_of_intervals]
list_of_days_tasks = list()
today_list = [0 for i in range(24)]
previous_date = "2009-06-10"
with open("./dataset/processed_data.txt", "w") as processed_data:
    with open("./dataset/data-2residents.csv") as input_file:
        input_data = csv.reader(input_file, delimiter=",")

        for line in input_data:
            date = line[0]
            time = int(line[1].split(":")[0])
            task = line[4]
            for i in list_of_intervals:
                dic_of_tasks = list_of_time_tasks[i]
                if (time >= i) and (time < i + len_intervals):
                    if task in dic_of_tasks:
                        dic_of_tasks[task] += 1
                    else:
                        dic_of_tasks[task] = 1

                    if date != previous_date:
                        list_of_days_tasks.append(today_list)
                        today_list = [0 for i in range(24)]

                    today_list[i] += 1

                    previous_date = date

    list_of_days_tasks.append(today_list)

    for i in list_of_time_tasks:
        processed_data.write(str(i) + "\n")

    print(list_of_days_tasks)
    print(len(list_of_days_tasks))
