import tensorflow as tf
from tensorflow import keras
import TokenParser

TokenParser.process_files("Tokens.dat",  "TokenfileOnlyTokens.log")
TokenParser.process_files("logfile.log", "LogfileOnlyTokens.log")

result = TokenParser.CrossReferenceTokens("LogfileOnlyTokens.log", "TokenfileOnlyTokens.log")
print("Cross reference passed = ", result)


