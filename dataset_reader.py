import csv
from setting import len_intervals


with open("./dataset/house1.csv", "w") as time_task:
    with open("./dataset/house1_raw_data.csv") as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        tasks = {1: "Leaving the house", 2: "", 3: "", 4: "Use toilet", 5: "Take shower", 6: "Brush teeth", 7: "",
                 8: "", 9: "Shave", 10: "Go to bed", 11: "Get dressed", 12: "", 13: "Prepare brunch",
                 14: "Prepare dinner", 15: "", 16: "", 17: "Get a drink", 18: "", 19: "", 20: "", 21: "", 22: "",
                 23: "", 24: "Wash dishes", 25: "", 26: "", 27: "", 28: "", 29: "Answer the phone", 30: "",
                 31: "Eat dinner", 32: "Eat brunch", 33: "Set up sensors", 34: "Unpack", 35: "Install sensor",
                 36: "On phone", 37: "Fasten kitchen camera", 38: "Wash toaster", 39: "", 40: "Play piano", 41: "",
                 42: "Search keys", 43: "Prepare for leaving", 44: "Drop dish", 45: "Water baobab"}
        for row in data:
            date = row[0]
            time = row[1]
            task = tasks[int(row[4])]
            if task != "":
                time_task.write(date + "," + time + "," + "," + "," + task + "\n")
