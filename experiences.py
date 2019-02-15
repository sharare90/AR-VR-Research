import math


class Experience:
    def __init__(self):
        self.valid_experiences = ["weather", "traffic", "news", "parking", "messages", "calendar"]
        self.max_value = 100
        self.experience_quality = 0
        self.delay = 0
        self.formats = ["4K", "HD", "Low-res", "animation"]
        self.current_format = ""

    def get_experience(self, exp_format):
        self.current_format = exp_format
        self.delay += 1

    def get_experience_score(self):
        scaling_factor = 1.0
        if self.current_format == "4K":
            scaling_factor = 1.0
        elif self.current_format == "HD":
            scaling_factor = 0.8
        elif self.current_format == "Low-res":
            scaling_factor = 0.5
        elif self.current_format == "animation":
            scaling_factor = 0.4
        return self.max_value * scaling_factor * math.exp(-0.9 * self.delay)
