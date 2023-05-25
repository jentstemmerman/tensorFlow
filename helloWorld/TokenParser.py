import re

def process_files(input_file, output_file):
    reading_file      = open(input_file,  "r")
    LogfileOnlyTokens = open(output_file, "w")

    for line in reading_file:
        line = line.strip()
        regular_expression_tokens = ".{1,2} .{1,2} .{1,2} .{1,2}"
        line = line[0:20]
        
        token_pattern = re.compile(regular_expression_tokens)
        stripped_line = re.search(token_pattern, line)

        if  (stripped_line):
            stripped_line_string = stripped_line.group()
            stripped_line_string = stripped_line_string.replace(" ", "")
            stripped_line_string = stripped_line_string + ","
            LogfileOnlyTokens.write(stripped_line_string)
        
    reading_file.close()
    LogfileOnlyTokens.close()

def CrossReferenceTokens(logFile, TokenFile):
    total = 0
    result = 1
    log_File    = open(logFile, "r")
    Token_File = open(TokenFile, "r")

    TokenFileString = Token_File.read()
    for line in log_File:
        line = line.strip()
        if (TokenFileString.find(line) == -1):
            print("Token " + line + "not found in tokenfile")
            result = -1
        total = total + 1
    return result



