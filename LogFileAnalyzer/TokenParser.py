import re
from pathlib import Path
import sys
import os

def GetItemListToSearchFor(RegexFile):
    RegexFile  = open(RegexFile, "r")
    line       = RegexFile.readline()
    lineNumber = 0
    itemList   = []

    while((line is not None) and ( line != '')):
        try:
            #line = line.group()
            compiledRegex = re.compile(line)
            itemList.append(compiledRegex)
        except:
            print("Error in RegexFile ", RegexFile, "at line ", lineNumber)
            break
        line        = RegexFile.readline()
        lineNumber += 1
    RegexFile.close()

    return itemList

def GetNextItemInLine(line, regularExpressionList):
    stripped_line_string = None
    for regularExpression in regularExpressionList:
        stripped_line = re.search(regularExpression, line)
        if  (stripped_line):
            stripped_line_string = stripped_line.group()
    
    return stripped_line_string


def getLinesAndTokens(itemList, numberOfTokens, reading_file, tokenList=None, PrintfList=None): 
    numberOfFoundTokens = 0
    
    if (tokenList is None):
        tokenList  = []

    if (PrintfList is None):
        PrintfList = []

    while ((numberOfTokens == -1) or (numberOfFoundTokens < numberOfTokens)):
        line = reading_file.readline()

        if (not line):
            break

        stripped_line = re.search(itemList[0], line)

        if  (stripped_line):
            numberOfFoundTokens += 1
            stripped_line_string = stripped_line.group()
            tokenList.append(stripped_line_string)
            PrintfList.append(line)
                
    return tokenList, PrintfList, numberOfFoundTokens

def process_files(input_file, output_file, itemList):
    if (not os.path.exists(output_file)):
        reading_file = open(input_file,  "r", errors='ignore')
        
        file = Path(output_file)
        file.parent.mkdir(parents=True, exist_ok=True)
        LogfileOnlyTokens = open(output_file, "w")
                                            
        tokenList, PrintfList, numberOfFoundTokens = getLinesAndTokens(itemList, -1, reading_file, tokenList=None, PrintfList=None)

        for line in tokenList:
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
