len_intervals = 1  # len_interval = 1 means that we have 24 intervals for a day. len_intervals = 2 means that we have 12 intervals.
house = "house2"
phase = "validation"
saved_model_folder = 'saved_models_d2'
num_train_days = 45  # for house1 = 17, for house2=45
num_test_days = 5  # for house1 = 2, for house2 = 5
num_validation_days = 5  # for house1 = 2, for house2 = 5
num_valid_requests = 13  # for house 1
# num_valid_requests = 13 #for house 2

# for creating requests num_valid_requests is 5
# weather, traffic, news, parking, recipe
# dict_reqs_nums = {'Summary of News': 0, 'Weather Report': 1, 'Parking status': 2, 'Traffic Report': 3, 'Recipe': 4}
# dict_reqs_nums = {
#     'Brush teeth': 0,
#     'Take shower': 1,
#     'Wash dishes': 2,
#     'Leaving the house': 3,
#     'Shave': 4,
#     'Prepare brunch': 5,
#     'Get a drink': 6,
#     'Prepare dinner': 7,
# }
# #
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

threshold_prob_based = 22  # dataset 1 = 22, dataset 2 = 28
caching_cost_each_hour = 1
no_respond = 3
number_of_caching_trial = 10

external_bandwidth = 100
delay_d = 0.99
size = 0.483 * 8
relative_quality = 0.81
delay = size / external_bandwidth
max_score = 1
