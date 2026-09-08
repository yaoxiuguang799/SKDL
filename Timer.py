import time

class Timer:
    def __init__(self, name=None, verbose=True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        if self.verbose:
            name = f"[{self.name}] " if self.name else ""
            print(f"{name}耗时: {self.elapsed:.6f} 秒")

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed