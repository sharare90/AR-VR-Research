from model import Environment, Server, User, Agent


env = Environment(time, devices)  # Environment
server = Server()
user = User(env)

agent = Agent(env, server, user)


while True:
    env.step()
    user.step()
