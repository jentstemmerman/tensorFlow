import sys
import os

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        directory = os.path.dirname(filename)

        if not os.path.exists(directory):
            os.makedirs(directory)

        self.logfile = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()