import re
from pathlib import Path
import os
import DataTreatment
import numpy as np

def GetItemListToSearchFor(RegexFile):
    import json
    itemList = None

    try:
        file = open(RegexFile, 'r')
        itemList = json.load(file)

    except Exception as e:
        print("Error in RegexFile :", e)
        exit()
          
    return itemList


def GetItemsInLine(line, itemList, dataTreatment):
    modified_line = line  # Create a copy of the original line
    items = []

    for index, item in enumerate(itemList):
        regularExpression = item["Regex"]
        stripped_line = re.search(regularExpression, modified_line)
        
        if (stripped_line is not None):
            result = stripped_line.group(1)
            modified_line = re.sub(regularExpression, '', modified_line)  # Remove the found regex
            result = dataTreatment.TreatValue(index, item, result, stripped_line)

            items.append(result)
        else:
            result = dataTreatment.TreatValue(index, item, "", stripped_line)
            items.append(result)

    return items




def getLinesAndTokens(itemList, numberOfTokens, reading_file=None, tokenList=None, PrintfList=None): 
    numberOfLines = 0
    
    if (tokenList is None):
        tokenList  = []

    if (PrintfList is None):
        PrintfList = []


    dataTreatment = DataTreatment.DataTreatment(len(itemList))

    while ((numberOfTokens == -1) or (numberOfLines < numberOfTokens)):
        line = reading_file.readline()
        lineBackup = line

        if (not line):
            break

        items = GetItemsInLine(line, itemList, dataTreatment)

        if (len(items) > 0):
            numberOfLines += 1
            tokenList.append(items)
            PrintfList.append(lineBackup)

    return tokenList, PrintfList, numberOfLines

def process_file_to_tokenFile(input_file, output_file, itemList):
    
    tokenListList = None
    reading_file = open(input_file,  "r", errors='ignore')
                                            
    tokenListList, _, _ = getLinesAndTokens(itemList, -1, reading_file, tokenList=None, PrintfList=None)

    if (output_file is not None):
        file = Path(output_file)
        file.parent.mkdir(parents=True, exist_ok=True)
        LogfileOnlyTokens = open(output_file, "w")
    
        lineNumber = 0
        stopLine = False
        for tokenList in tokenListList:
            line = ""
            for index, token in enumerate(tokenList):
                if index < len(tokenList) - 1:
                    line = line + str(token) + ","
                else:
                    line = line + str(token) 

        #     if (index == 1):  # vuile hack
        #         value = int()
        #         if(int(token) > 100000):
        #             print("stop at " + str(token) + " at line ", str(lineNumber))
        #             stopLine = True
        #             break

            if (stopLine):
                stopLine = False
                continue
            lineNumber += 1

            line = line + '\n'
            LogfileOnlyTokens.write(line)
        LogfileOnlyTokens.close()


    reading_file.close()

    return tokenListList


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

def split_input_target_multidim(sequence):
    # Tensorflow -> input_text: Tensorflo, target_text = ensorfloW
    input_text = sequence[:-1,:]
    target_text = sequence[1:,:]
    return input_text, target_text

def returnValue(sequence):
    return sequence




def process_log_files_to_tokens_recursive(directory, outputDirectory, itemList):
    tokenListListList = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if not root.endswith(os.path.sep + "out") and filename.endswith(".log"):
                full_path = os.path.join(root, filename)
                # Do something with the content of the log file
                outputFile = outputDirectory + "/" + filename.replace(".log", "") + "_onlyTokens.log" 
                #if (not os.path.exists(outputFile)):
                tokenListList = process_file_to_tokenFile(full_path, outputFile, itemList)
                tokenListListList.append(tokenListList)
    return tokenListListList

def TransformToListIfSingularType(value):
    isList = type(value) is list  
    isnumpyArray = type(value) is np.ndarray       
    if (not isList) and (not isnumpyArray):
        newList = []
        newList.append(value)
        value = newList

    return value
