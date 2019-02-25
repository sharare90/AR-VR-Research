from model import Environment, Server, User, Agent

env = Environment()  # Environment
server = Server()
user = User(env, "./dataset/house1/")

agent = Agent(env, server, user)

for _ in range(10000):
    env.step()
    user.step()
