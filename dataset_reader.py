import csv

with open("./dataset/house1.csv") as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    tasks = {1: "Leaving the house", 4: "Use toilet", 5: "Take shower", 6: "Brush teeth", 7: " ", 8: " ", 9: "Shave",
             10: "Go to bed", 11: "Get dressed", 12: " ", 13: "Prepare brunch", 14: "Prepare dinner", 17: "Get a drink",
             18: " ",
             20: " ", 23: " ", 24: "Wash dishes", 29: "Answer the phone", 31: "Eat dinner", 32: "Eat brunch",
             33: "Set up sensors", 34: "Unpack",
             35: "Install sensor", 36: "On phone", 37: "Fasten kitchen camera", 38: "Wash toaster", 40: "Play piano",
             42: "Search keys", 43: "Prepare for leaving", 44: "Drop dish", 45: "Water baobab"}
    for row in data:
        print("User " + tasks[int(row[4])] + " from " + row[1] + " on " + row[0] + " to " + row[3] + " on " + row[2])
