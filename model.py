class Environment:
    def __init__(self, time, devices):
        self.time = time
        self.devices = devices


class Server:
    def __init__(self):
        pass


class User:
    def __init__(self, env):
        self.env = env
        self.request = ""


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
