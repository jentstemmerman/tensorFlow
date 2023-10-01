import multiprocessing
import TokenParser
import time
import tensorflow as tf
import numpy as np


def ComposeLineList(numberOfParallelExecutions, nextPredictedTokens, nextTokens, PrintfList, probabilityOfNextTokens, probabilityPredictions):
    
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


class InferenceOutputClass:
    def __init__(self, numberOfParallelExecutions, nextPredictedTokens, nextTokens, PrintfList, probabilityOfNextTokens, probabilityPredictions):
        self.nextPredictedTokens = nextPredictedTokens
        self.nextTokens = nextTokens
        self.numberOfParallelExecutions = numberOfParallelExecutions
        self.PrintfList = PrintfList
        self.probabilityOfNextTokens = probabilityOfNextTokens
        self.probabilityPredictions = probabilityPredictions
        
        self.linesToWrite  = []
        self.numberOfIdenticalsInThisCycle = None
        self.accumulatedNextTokenProbability = None
        self.accumulatedPredictedTokenProbability = None
        
    
    def InferenceOutputThread(self):
        self.linesToWrite, self.numberOfIdenticalsInThisCycle, self.accumulatedNextTokenProbability, self.accumulatedPredictedTokenProbability = ComposeLineList(self.numberOfParallelExecutions, self.nextPredictedTokens, self.nextTokens, self.PrintfList, self.probabilityOfNextTokens, self.probabilityPredictions)


    def StartInferenceOutput(self):
        self.InferenceOutput_thread = multiprocessing.Process(target=self.InferenceOutputThread, args=())
        self.InferenceOutput_thread.start()

    def WaitForCompletion(self):
        # Wait for the thread to finish 
        self.InferenceOutput_thread.join()
        return self.linesToWrite, self.numberOfIdenticalsInThisCycle, self.accumulatedNextTokenProbability, self.accumulatedPredictedTokenProbability

