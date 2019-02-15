class User:
    def __init__(self, wake_time, bed_time, work_st, work_end):
        self.work_st_time = work_st
        self.work_en_time = work_end
        self.wake_time = wake_time
        self.bed_time = bed_time

    def user_request_policy(self):
        pass
