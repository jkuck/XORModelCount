__author__ = 'shengjia'

import threading
import time

class Timer:
    def __init__(self, max_time):
        self.max_time = max_time
        self.begin_time = time.time()
        self.time_out_flag = False
        self.time_out_flag_lock = threading.Lock()

        self.timer_thread = threading.Thread(target=self.timer)
        self.timer_thread.start()

    def __del__(self):
        try:
            self.begin_time = 0.0
            self.timer_thread.join()
        except:
            pass

    # Timer thread that wakes up every second to check if the specified time has elapsed
    def timer(self):
        while True:
            if time.time() - self.begin_time > self.max_time:
                break
            time.sleep(1.0)
        self.time_out_flag_lock.acquire()
        self.time_out_flag = True
        self.time_out_flag_lock.release()

    # Query of whether the timer has timed out
    def timeout(self):
        self.time_out_flag_lock.acquire()
        if self.time_out_flag:
            self.time_out_flag_lock.release()
            return True
        else:
            self.time_out_flag_lock.release()
            return False

    # Remaining time before timeout
    def time(self):
        remained = self.max_time - (time.time() - self.begin_time)
        if remained < 0:
            return 0
        else:
            return remained
