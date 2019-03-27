len_intervals = 1
house = "house2"
phase = "test"
num_valid_requests = 13  # weather, traffic, news, parking, recipe
# dict_reqs_nums = {'Summary of News': 0, 'Weather Report': 1, 'Parking status': 2, 'Traffic Report': 3, 'Recipe': 4}
dict_reqs_nums = {
    'R1 sleep': 0,
    'R2 sleep': 1,
    'Dinner': 2,
    'Lunch': 3,
    'Leave home': 4,
    'Breakfast': 5,
    'Laundry': 6,
    'R1 wake': 7,
    'R2 wake': 8,
    'R2 take medicine': 9,
    'R1 work in office': 10,
    'Night wandering': 11,
    'Bed to toilet': 12,
}
num_actions = 13  # for dataset1 = 8, and for dataset2 = 13

cashing_cost_each_hour = 1
respond_reward = 2
number_of_cashing_trial = 1000
