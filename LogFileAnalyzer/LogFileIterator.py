import os

class FileIterator():
    
    def __init__(self, directory, extention):
        self.directory = directory
        self.counter = 0
        self.extention = extention
        
    def get_first_file(self):
        self.counter = 0
        return self.get_file(self.counter)

    def get_next_file(self):
        self.counter = self.counter + 1
        return self.get_file(self.counter)

    def get_file(self, requested_counter):
        counter = 0
        for root, dirs, files in os.walk(self.directory):
            for filename in files:
                if filename.endswith(self.extention):
                    if (counter == requested_counter):
                        full_path = os.path.join(root, filename)
                        outputFile = full_path
                        return outputFile
                    counter = counter + 1
                    
        return None