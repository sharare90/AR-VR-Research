from setting import house, phase, caching_cost_each_hour, respond_reward, num_valid_requests, dict_reqs_nums, \
    number_of_caching_trial
import numpy as np


class Caching(object):
    def __init__(self):
        self.data_file = "./dataset/" + house + "/LSTM_input_" + phase + "_req.npy"
        self.data = np.load(self.data_file)

    def random_caching(self):
        average_cost = 0

        for i in range(number_of_caching_trial):
            cost = 0
            for i in self.data:
                for j in range(num_valid_requests):
                    random_action_num = np.random.randint(0, 2)
                    if i[j] == random_action_num:
                        cost -= respond_reward
                    else:
                        cost += caching_cost_each_hour
            average_cost += cost
        average_cost /= number_of_caching_trial
        return average_cost

    def cach_everything(self):
        cost = 0
        for i in self.data:
            for j in range(num_valid_requests):
                if i[j] == 1:
                    cost -= respond_reward
                else:
                    cost += caching_cost_each_hour
        return cost

    def average_based_caching(self):
        pass

    def LSTM_based_caching(self):
        pass

cache = Caching()
random_cost = cache.random_caching()
cach_all_cost = cache.cach_everything()
print(random_cost)
print(cach_all_cost)
