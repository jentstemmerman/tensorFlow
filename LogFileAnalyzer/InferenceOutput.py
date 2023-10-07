import multiprocessing
import TokenParser
import time
import tensorflow as tf
import numpy as np


def ComposeLineList(threadId, numberOfParallelExecutions, nextPredictedTokens, nextTokens, PrintfList, probabilityOfNextTokens, probabilityPredictions):
    with tf.device('/CPU:0'):
        linesToWrite = []
        numberOfIdenticals = 0
        accumulatedNextTokenProbability = 0
        accumulatedPredictedTokenProbability = 0

        for executionIndex in range(numberOfParallelExecutions):
            nextToken = nextTokens[executionIndex] 
            nextPredictedToken = nextPredictedTokens[executionIndex].decode('utf-8')  
            
            if (nextToken == nextPredictedToken):
                nextPredictedToken = "Identical  "
                numberOfIdenticals += 1

            nextTokenProbability = (probabilityOfNextTokens[executionIndex]).numpy()
            lineToWrite = "Actual: " + nextToken  + "  {:2.3f}".format(nextTokenProbability) + ". Pred: " + nextPredictedToken + " {:2.3f}".format(probabilityPredictions[executionIndex].numpy()) + ". " + PrintfList[executionIndex]
            linesToWrite.append(lineToWrite)
            accumulatedNextTokenProbability += nextTokenProbability
            accumulatedPredictedTokenProbability += probabilityPredictions[executionIndex].numpy()
        return linesToWrite, numberOfIdenticals, accumulatedNextTokenProbability, accumulatedPredictedTokenProbability


      
    
def InferenceOutputThread(threadId, numberOfParallelExecutions, nextPredictedTokens, nextTokens, PrintfList, probabilityOfNextTokens, probabilityPredictions):
    with tf.device('/CPU:0'):
        print("start InferenceOutputThread", threadId)
        linesToWrite, numberOfIdenticalsInThisCycle, accumulatedNextTokenProbability, accumulatedPredictedTokenProbability = ComposeLineList(threadId, numberOfParallelExecutions, nextPredictedTokens, nextTokens, PrintfList, probabilityOfNextTokens, probabilityPredictions)
        print("end InferenceOutputThread")
        return linesToWrite, numberOfIdenticalsInThisCycle, accumulatedNextTokenProbability, accumulatedPredictedTokenProbability


