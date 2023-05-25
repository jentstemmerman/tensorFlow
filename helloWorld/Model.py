import tensorflow as tf
from tensorflow import keras
import TokenParser

TokenFileName = "Tokens.dat"
LogFileName   = "logfile.log"

TokenFileNameCvs = "TokenfileOnlyTokens.log"
LogFileNameCvs   = "LogfileOnlyTokens.log"

TokenParser.process_files(TokenFileName,  TokenFileNameCvs)
TokenParser.process_files(LogFileName, LogFileNameCvs)

result = TokenParser.CrossReferenceTokens(LogFileNameCvs, TokenFileNameCvs)
print("Cross reference passed = ", result)


