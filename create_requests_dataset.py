import numpy as np
import csv


def action_to_request_house1(action):
    request = ""
    if action == "Shave" or action == "Brush teeth" or action == "Get a drink":
        request = "Summary of News"
    elif action == "Get dressed":
        request = "Weather Report"

    elif action == "Prepare for leaving":
        r = np.random.uniform(0, 1)
        if r > 0.5:
            request = "Traffic Report"
        elif r > 0.2:
            request = "Weather Report"
        else:
            request = "Parking status"
    elif action == "Prepare brunch" or action == "Prepare dinner":
        request = "Recipe"

    return request


def action_to_request_house2(action):
    request = ""
    if action == "R1 wake" or action == "R2 wake":
        request = "Summary of News"
    elif action == "Breakfast":
        r = np.random.uniform(0, 1)
        if r > 0.7:
            request = "Weather Report"
        else:
            request = "Recipe"
    elif action == "Leave home":
        r = np.random.uniform(0, 1)
        if r > 0.5:
            request = "Traffic Report"
        elif r > 0.2:
            request = "Weather Report"
        else:
            request = "Parking status"
    elif action == "Lunch" or action == "Dinner":
        request = "Recipe"

    return request


with open("./dataset/house_acts_reqs_train.csv", "w") as house_act_req_tr:
    with open("./dataset/house1/generated_data.csv") as house1_gen_data:
        data1 = csv.reader(house1_gen_data, delimiter=',')
        for line in data1:
            time = line[0]
            action = line[1]
            request = action_to_request_house1(action)
            if request != "":
                house_act_req_tr.write("house1," + time + "," + action + "," + request + "\n")
    with open("./dataset/house2/generated_data.csv") as house2_gen_data:
        data2 = csv.reader(house2_gen_data, delimiter=',')
        for line in data2:
            time = line[0]
            action = line[1]
            request = action_to_request_house2(action)
            if request != "":
                house_act_req_tr.write("house2," + time + "," + action + "," + request + "\n")

with open("./dataset/house_acts_reqs_test.csv", "w") as house_act_req_te:
    with open("./dataset/house1/time_task.csv") as house1_gen_data:
        data1 = csv.reader(house1_gen_data, delimiter=',')
        for line in data1:
            time = line[0]
            action = line[1]
            request = action_to_request_house1(action)
            if request != "":
                house_act_req_te.write("house1," + time + "," + action + "," + request + "\n")
    with open("./dataset/house2/time_task.csv") as house2_gen_data:
        data2 = csv.reader(house2_gen_data, delimiter=',')
        for line in data2:
            time = line[0]
            action = line[1]
            request = action_to_request_house2(action)
            if request != "":
                house_act_req_te.write("house2," + time + "," + action + "," + request + "\n")
