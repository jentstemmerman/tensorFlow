import tensorflow as tf
from tensorflow import keras
import TokenParser
import numpy as np

TokenFileName = "Tokens.dat"
LogFileName   = "logfile.log"

TokenFileNameCvs = "TokenfileOnlyTokens.log"
LogFileNameCvs   = "LogfileOnlyTokens.log"

TokenParser.process_files(TokenFileName,  TokenFileNameCvs)
TokenParser.process_files(LogFileName, LogFileNameCvs)

result = TokenParser.CrossReferenceTokens(LogFileNameCvs, TokenFileNameCvs)
print("Cross reference passed = ", result)

if (result):
    print("Reading dictionary into memory, not sure I will need it")
    # using loadtxt()
    tokenNumpyList = np.loadtxt(TokenFileNameCvs, delimiter=",", dtype=str)
    tokenList      = tf.convert_to_tensor(tokenNumpyList)
    print("Reading dictionary into memory DONE")

    print("Reading logfile into memory, might want to improve this later")
    logTokenNumpyList = np.loadtxt(LogFileNameCvs, delimiter=",", dtype=str)
    logtokenList      = tf.convert_to_tensor(logTokenNumpyList)

    print("The Shape of tokenList tensor is",tf.shape(tokenList))
    print("The Shape of logtokenList tensor is",tf.shape(logtokenList))
    print("Reading logfile into memory DONE")
    
    # encoder
    ids_from_token   = tf.keras.layers.StringLookup(vocabulary=tokenList, mask_token=None)
    # decoder
    token_from_id    = tf.keras.layers.StringLookup(vocabulary=tokenList, mask_token=None, invert=True)

    ids_from_logfile = ids_from_token(logtokenList)
    print("ids_from_logfile:")
    print(ids_from_logfile)
    print("token_from_id:")
    print(token_from_id(ids_from_logfile))

    print("END")