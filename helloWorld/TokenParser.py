import re
from pathlib import Path

def process_files(input_file, output_file):
    reading_file      = open(input_file,  "r", errors='ignore')
    
    file = Path(output_file)
    file.parent.mkdir(parents=True, exist_ok=True)
    LogfileOnlyTokens = open(output_file, "w")

    for line in reading_file:
        line = line.strip()
        regular_expression_tokens = ".{1,2} .{1,2} .{1,2} .{1,2}"
        line = line[0:20]
        
        token_pattern = re.compile(regular_expression_tokens)
        stripped_line = re.search(token_pattern, line)

        if  (stripped_line):
            stripped_line_string = stripped_line.group()
            stripped_line_string = stripped_line_string + '\n'
            LogfileOnlyTokens.write(stripped_line_string)
        
    reading_file.close()
    LogfileOnlyTokens.close()

def CrossReferenceTokens(logFile, TokenFile):
    total = 0
    result = 1
    log_File   = open(logFile, "r")
    Token_File = open(TokenFile, "r")

    TokenFileString = Token_File.read()
    for line in log_File:
        line = line.strip()
        if (TokenFileString.find(line) == -1):
            print("Token " + line + "not found in tokenfile")
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



def GetTokenTranslation(token, TokenFile):
    # Tensorflow -> input_text: Tensorflo, target_text = ensorfloW
    Token_File = open(TokenFile, "r")
    TokenFileLines = Token_File.readlines()
    TranslationString = ""

    for line in TokenFileLines:
        if (line.find(token) != -1):
            TranslationString = line
            break
    
    return TranslationString

#def AddLogFileToDataSet(input_file):



