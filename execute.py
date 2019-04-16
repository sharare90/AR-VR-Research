from model import Environment, Server, User, Agent
from setting import house, len_intervals, phase, num_actions, dict_reqs_nums

file_name = house

env = Environment()  # Environment
server = Server()
user = User(env, "./dataset/" + file_name + "/")

agent = Agent(env, server, user)

q = [7] * num_actions

with open("./dataset/" + file_name + "/generated_data_" + phase + ".csv", "w") as gen_data:
    for _ in range(10000):
        env.step()
        time, interval_actions = user.step()

        interval = int(time.time().hour / len_intervals)

        for act in interval_actions:
            if file_name == "house2":
                act = act[:-6]

            if q[dict_reqs_nums[act]] < 6:
                continue

            q[dict_reqs_nums[act]] = 0

            gen_data.write(str(interval) + "," + act + "\n")

        for i in range(len(q)):
            q[i] += 1
