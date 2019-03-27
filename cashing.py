from setting import house, phase, cashing_cost_each_hour, respond_reward, num_valid_requests, dict_reqs_nums, \
    number_of_cashing_trial
import numpy as np


class Cashing(object):
    def __init__(self):
        self.data_file = "./dataset/" + house + "/LSTM_input_" + phase + "_req.npy"
        self.data = np.load(self.data_file)

    def random_cashing(self):
        average_cost = 0

        for i in range(number_of_cashing_trial):
            cost = 0
            for i in self.data:
                for j in range(num_valid_requests):
                    random_action_num = np.random.randint(0, 2)
                    if i[j] == random_action_num:
                        cost -= respond_reward
                    else:
                        cost += cashing_cost_each_hour
            average_cost += cost
        average_cost /= number_of_cashing_trial
        return average_cost

    def cash_everything(self):
        cost = 0
        for i in self.data:
            for j in range(num_valid_requests):
                if i[j] == 1:
                    cost -= respond_reward
                else:
                    cost += cashing_cost_each_hour
        return cost

    def average_based_cashing(self):
        pass


cash = Cashing()
random_cost = cash.random_cashing()
cash_all_cost = cash.cash_everything()
print(random_cost)
print(cash_all_cost)
