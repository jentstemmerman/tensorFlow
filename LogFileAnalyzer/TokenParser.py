import re
from pathlib import Path
import sys
import os

def GetItemListToSearchFor(RegexFile):
    import json
    itemList = None

    try:
        file = open(RegexFile, 'r')
        itemList = json.load(file)
        print(itemList)
    except Exception as e:
        print("Error in RegexFile :", e)
        exit()
          
    return itemList


def GetItemsInLine(line, itemList):
    modified_line = line  # Create a copy of the original line
    items = []

    for item in itemList:
        regularExpression = item["Regex"]
        stripped_line = re.search(regularExpression, modified_line)
        
        if stripped_line:
            stripped_string = stripped_line.group()
            modified_line = re.sub(regularExpression, '', modified_line)  # Remove the found regex
            items.append(stripped_string)

    return items


def getLinesAndTokens(itemList, numberOfTokens, reading_file, tokenList=None, PrintfList=None): 
    numberOfFoundItems = 0
    
    if (tokenList is None):
        tokenList  = []

    if (PrintfList is None):
        PrintfList = []

    while ((numberOfTokens == -1) or (numberOfFoundItems < numberOfTokens)):
        line = reading_file.readline()
        lineBackup = line

        if (not line):
            break

        items = GetItemsInLine(line, itemList)

        if (len(items) > 0):
            numberOfFoundItems += len(items)
            tokenList.append(items)
            PrintfList.append(lineBackup)

    return tokenList, PrintfList, numberOfFoundItems

def process_files(input_file, output_file, itemList):
    if (not os.path.exists(output_file)):
        reading_file = open(input_file,  "r", errors='ignore')
        
        file = Path(output_file)
        file.parent.mkdir(parents=True, exist_ok=True)
        LogfileOnlyTokens = open(output_file, "w")
                                            
        tokenListList, PrintfList, numberOfFoundTokens = getLinesAndTokens(itemList, -1, reading_file, tokenList=None, PrintfList=None)

        for tokenList in tokenListList:
            line = ""
            for token in tokenList:
                line = line + token
            line = line + '\n'
            LogfileOnlyTokens.write(line)

        reading_file.close()
        LogfileOnlyTokens.close()


def CrossReferenceTokens(directory, TokenFile):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".log"):
                full_path = os.path.join(root, filename)
                total = 0
                result = 1
                log_File   = open(full_path, "r")
                Token_File = open(TokenFile, "r")

                TokenFileString = Token_File.read()
                for line in log_File:
                    line = line.strip()
                    if (TokenFileString.find(line) == -1):
                        print("Token " + line + " not found in tokenfile")
                        result = -1
                    total = total + 1
    return result


def split_input_target(sequence):
    # Tensorflow -> input_text: Tensorflo, target_text = ensorfloW
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

def returnValue(sequence):
    return sequence

def process_log_files_to_tokens_recursive(directory, outputDirectory, itemFile):
    itemList = GetItemListToSearchFor(itemFile)
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if not root.endswith(os.path.sep + "out") and filename.endswith(".log"):
                full_path = os.path.join(root, filename)
                # Do something with the content of the log file
                outputFile = outputDirectory + "/" + filename.replace(".log", "") + "_onlyTokens.log" 
                if (not os.path.exists(outputFile)):
                    process_files(full_path, outputFile, itemList)


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

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()
