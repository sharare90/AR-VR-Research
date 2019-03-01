import csv
import numpy as np

with open("./dataset/train_house2.csv", "w") as train_h1:
    with open("./dataset/house2.csv") as h1:
        data1 = csv.reader(h1, delimiter=",")
        dates = {}
        list_of_days_data = list()
        day_counter = 1
        for line in data1:
            date = line[0]
            if date not in dates.values():
                list_of_one_day_data = list()
                dates[day_counter] = date
                day_counter += 1
        validation_days = np.random.randint(0, day_counter, int(0.2 * day_counter))

