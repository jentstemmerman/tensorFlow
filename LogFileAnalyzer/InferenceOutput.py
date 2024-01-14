import multiprocessing
import TokenParser
import time
import tensorflow as tf
import numpy as np

def ComposeLineList(numberOfParallelExecutions, nextPredictedTokens, nextTokens, PrintfList, probabilityOfNextTokens, probabilityPredictions):
    with tf.device('/CPU:0'):
        linesToWrite = []
        numberOfIdenticals = 0
        accumulatedNextTokenProbability = 0
        accumulatedPredictedTokenProbability = 0

        for executionIndex in range(numberOfParallelExecutions):
            nextToken = nextTokens[executionIndex] 
            nextPredictedToken = nextPredictedTokens[executionIndex].decode('utf-8')  
            
            if (nextToken[0] == nextPredictedToken):
                nextPredictedToken = "Identical  "
                numberOfIdenticals += 1

            nextTokenProbabilities = (probabilityOfNextTokens[executionIndex]).numpy()
            nextTokenProbabilities = TokenParser.TransformToListIfSingularType(nextTokenProbabilities)
            #nextTokenProbability = (probabilityOfNextTokens[executionIndex]).numpy()
            lineToWrite = "Actual: " + nextToken[0]  + "".join("  {:2.3f}".format(nextTokenProbability) for nextTokenProbability in nextTokenProbabilities) + \
                ". Pred: " + nextPredictedToken + " {:2.3f}".format(probabilityPredictions[executionIndex].numpy()) + ". " + PrintfList[executionIndex] + "<br>" 
            linesToWrite.append(lineToWrite)

            for nextTokenProbability in nextTokenProbabilities:
                accumulatedNextTokenProbability += nextTokenProbability

            accumulatedPredictedTokenProbability += probabilityPredictions[executionIndex].numpy()
        return linesToWrite, numberOfIdenticals, accumulatedNextTokenProbability, accumulatedPredictedTokenProbability




def ComposeLineListOld(numberOfParallelExecutions, nextPredictedTokens, nextTokens, PrintfList, probabilityOfNextTokens, probabilityPredictions):
    with tf.device('/CPU:0'):
        linesToWrite = []
        numberOfIdenticals = 0
        accumulatedNextTokenProbability = 0
        accumulatedPredictedTokenProbability = 0

        for executionIndex in range(numberOfParallelExecutions):
            nextToken = nextTokens[executionIndex] 
            nextPredictedToken = nextPredictedTokens[executionIndex].decode('utf-8')  
            
            if (nextToken == nextPredictedToken):
                nextPredictedToken = '<span style="color:green;">Identical  </span>   '
                numberOfIdenticals += 1
            else:
                nextPredictedToken = '<span style="color:red;">' +  nextPredictedToken + '</span>'

            nextTokenProbabilities = (probabilityOfNextTokens[executionIndex]).numpy()
            nextTokenProbabilities = TokenParser.TransformToListIfSingularType(nextTokenProbabilities)

            lineToWrite = "Actual: " + nextToken[0]  + "".join("  {:2.3f}".format(nextTokenProbability) for nextTokenProbability in nextTokenProbabilities) + \
                ". Pred: " + nextPredictedToken + " {:2.3f}".format(probabilityPredictions[executionIndex].numpy()) + ". " + PrintfList[executionIndex] + "<br>" 
            linesToWrite.append(lineToWrite)

            for nextTokenProbability in nextTokenProbabilities:
                accumulatedNextTokenProbability += nextTokenProbability

            accumulatedPredictedTokenProbability += probabilityPredictions[executionIndex].numpy()
        return linesToWrite, numberOfIdenticals, accumulatedNextTokenProbability, accumulatedPredictedTokenProbability
     
    
def InferenceOutputThread(workerParameter, threadId, numberOfParallelExecutions, nextPredictedTokens, nextTokens, PrintfList, probabilityOfNextTokens, probabilityPredictions):
    with tf.device('/CPU:0'):
        linesToWrite, numberOfIdenticalsInThisCycle, accumulatedNextTokenProbability, accumulatedPredictedTokenProbability = ComposeLineList(numberOfParallelExecutions, nextPredictedTokens, nextTokens, PrintfList, probabilityOfNextTokens, probabilityPredictions)
        return linesToWrite, numberOfIdenticalsInThisCycle, accumulatedNextTokenProbability, accumulatedPredictedTokenProbability


