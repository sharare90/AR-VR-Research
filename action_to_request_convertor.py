import numpy as np


def action_to_request(action):
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
