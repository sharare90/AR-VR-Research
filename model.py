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
    def __init__(self, action_prob, mean, std):
        self.action_prob = action_prob
        self.mean = mean
        self.std = std

    def sample(self):
        estimate_num_actions = np.random.normal(self.mean, self.std)
        prob = estimate_num_actions - int(estimate_num_actions)
        num_actions = int(estimate_num_actions)
        if np.random.uniform(0, 1) < prob:
            num_actions += 1

        sampled_actions = []
        for i in range(num_actions):
            random_value = np.random.uniform(0, 1)
            for j in self.action_prob:
                if random_value - j[1] < 0:
                    sampled_action = j[0]
                    if sampled_action in sampled_actions:
                        continue
                    sampled_actions.append(sampled_action)
                else:
                    random_value = random_value - j[1]

        return sampled_actions


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
        interval_actions = self.intervals_data[interval_index].sample()
        print('{time}: {actions}'.format(time=current_time, actions=interval_actions))

    def process(self):
        daily_tasks = os.path.join(self.user_profile_dir, 'daily_tasks.txt')
        num_actions_per_interval = []
        with open(daily_tasks) as daily_tasks_info:
            for line in daily_tasks_info:
                line = line[1:-2].split(",")
                data = list(map(int, line))
                num_actions_per_interval.append(data)

        num_actions_per_interval = np.array(num_actions_per_interval)

        action_probs_file = os.path.join(self.user_profile_dir, 'processed_data.txt')
        action_probs = []

        interval_counter = 0
        with open(action_probs_file) as action_probs_info:
            for line in action_probs_info:
                interval_action = []
                line = line[1: -2].split(",")
                num_actions_in_current_interval = np.sum(num_actions_per_interval[:, interval_counter])
                interval_counter += 1

                for data in line:
                    data = data.split(":")
                    action_name = data[0].strip()[1:-1]
                    if 'end' in action_name:
                        continue

                    interval_action.append((action_name, float(data[1]) / num_actions_in_current_interval))
                    interval_action = sorted(interval_action, key=lambda x: x[1], reverse=True)
                action_probs.append(interval_action)

        intervals_data = []
        for i in range(12):
            mean = np.mean(num_actions_per_interval[:, i])
            std = np.std(num_actions_per_interval[:, i])
            interval_data = IntervalData(action_prob=action_probs[i], mean=mean, std=std)
            intervals_data.append(interval_data)

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
