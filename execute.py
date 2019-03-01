from model import Environment, Server, User, Agent


file_name = "house2"

env = Environment()  # Environment
server = Server()
user = User(env, "./dataset/" + file_name + "/")

agent = Agent(env, server, user)

with open("./dataset/" + file_name + "/generated_data.csv", "w") as gen_data:
    for _ in range(10000):
        env.step()
        time, interval_actions = user.step()
        for act in interval_actions:
            if file_name == "house2":
                gen_data.write(str(int(time.time().hour / 2)) + "," + act[:-6] + "\n")
            else:
                gen_data.write(str(int(time.time().hour / 2)) + "," + act + "\n")
