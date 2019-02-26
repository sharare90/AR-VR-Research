from model import Environment, Server, User, Agent

env = Environment()  # Environment
server = Server()
user = User(env, "./dataset/house1/")

agent = Agent(env, server, user)

with open("./dataset/house1/generated_data.csv", "w") as gen_data:
    for _ in range(10000):
        env.step()
        time, interval_actions = user.step()
        for act in interval_actions:
            gen_data.write(str(int(time.time().hour / 2)) + "," + act[:-6] + "\n")
