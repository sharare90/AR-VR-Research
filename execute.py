from model import Environment, Server, User, Agent

env = Environment()  # Environment
server = Server()
user = User(env, "./dataset/house1/")

agent = Agent(env, server, user)

while True:
    env.step()
    user.step()
