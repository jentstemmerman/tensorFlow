import os

class LogfileIterator():
    def __init__(self, directory):
        self.directory = directory
        self.counter = 0

    def get_first_tokenlogfile(self):
        self.counter = 0
        return self.get_tokenlogfile(self.counter)

    def get_next_tokenlogfile(self):
        self.counter = self.counter + 1
        return self.get_tokenlogfile(self.counter)

    def get_tokenlogfile(self, requested_counter):
        counter = 0
        for root, dirs, files in os.walk(self.directory):
            for filename in files:
                if filename.endswith(".log"):
                    if (counter == requested_counter):
                        full_path = os.path.join(root, filename)
                        outputFile = full_path
                        return outputFile
                    counter = counter + 1
                    
        return None