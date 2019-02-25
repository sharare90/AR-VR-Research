import os
import numpy as np
from datetime import datetime, timedelta


class Environment:
    def __init__(self):
        self.time = datetime(
            year=2000,
            month=1,
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0
        )

    def step(self):
        self.time = self.time + timedelta(hours=2)

    def get_time(self):
        return self.time


class Server:
    def __init__(self):
        pass


class IntervalData(object):
    def __init__(self, actions, mean, std):
        self.actions = actions
        self.mean = mean
        self.std = std

    def sample(self):
        #  num_acgions = np.normal.random()
        # for i in range(num_actions):
        # sample from self.actions
        pass


class User:
    def __init__(self, env, user_profile_dir):
        self.env = env
        self.request = ""
        self.user_profile_dir = user_profile_dir
        self.intervals_data = self.process()
        print("test")

    def step(self):
        current_time = self.env.get_time()
        interval_index = int(current_time.hour / 2)
        self.intervals_data[interval_index].sample()

    def process(self):
        daily_tasks = os.path.join(self.user_profile_dir, 'daily_tasks.txt')
        num_actions_per_interval = []
        with open(daily_tasks) as daily_tasks_info:
            for line in daily_tasks_info:
                line = line[1:-2].split(",")
                data = list(map(int, line))
                num_actions_per_interval.append(data)

        action_probs_file = os.path.join(self.user_profile_dir, 'processed_data.txt')
        action_probs = []
        with open(action_probs_file) as action_probs_info:
            for line in action_probs_info:
                interval_dict = {}
                line = line[1: -2].split(",")
                for data in line:
                    data = data.split(":")
                    interval_dict[data[0].strip()[1:-1]] = int(data[1])

                action_probs.append(interval_dict)

        intervals_data = []
        for i in range(12):
            #  create interval data from these two:
            #  action_probs[i]  # calculate probabilities of selecting each action
            #  num_actions_per_interval[:, i]  # calculate mean and std
            # interval_data = IntervalData(mean=, std, actions=)
            # intervals_data.append(interval_data)
            pass

        return intervals_data


class Scenarios:
    def __init__(self, user, env):
        self.user = user
        self.env = env

    def gen_scenarios(self):
        pass


class Agent:
    def __init__(self, env, server, user):
        self.env = env
        self.server = server
        self.user = user

    def predict_request(self):
        pass
