import csv
import random
from setting import house

with open("./dataset/" + house + "validation_" + house + ".csv", "w") as validation_h1:
    with open("./dataset/house2.csv") as h1:
        data1 = csv.reader(h1, delimiter=",")
        dates = {}
        one_day_data = list()
        current_date = "2009-06-10"
        for line in data1:
            date = line[0]
            if date != current_date:
                current_date = date
                dates[date] = one_day_data
                one_day_data = list()

            one_day_data.append(line)

        validation_data = dict(random.sample(dates.items(), int(0.1 * len(dates))))
        for i in validation_data:
            for j in dates[i]:
                validation_h1.write(j[0] + "," + j[1] + "," + j[2] + "," + j[3] + "," + j[4] + "\n")

        for i in validation_data:
            del dates[i]

with open("./dataset/test_" + house + ".csv", "w") as test_h1:
    test_data = dict(random.sample(dates.items(), int(0.1 * len(dates))))

    for i in test_data:
        for j in dates[i]:
            test_h1.write(j[0] + "," + j[1] + "," + j[2] + "," + j[3] + "," + j[4] + "\n")

    for i in test_data:
        del dates[i]

with open("./dataset/train_" + house + ".csv", "w") as train_h1:

    for i in dates.values():
        for j in i:
            train_h1.write(j[0] + "," + j[1] + "," + j[2] + "," + j[3] + "," + j[4] + "\n")
