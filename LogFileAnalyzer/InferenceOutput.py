import multiprocessing
import TokenParser
import time
import tensorflow as tf
import numpy as np
from numbers import Number

def TranslateToString(value):
    if isinstance(value, (Number, np.number)):
        value = str(value)
    else:
        value = value
    return value

def ComposeLineList(numberOfParallelExecutions, id_from_tokenList, predictions, actuals, PrintfList, probabilityOfActuals, probabilityOfPredictions):
    with tf.device('/CPU:0'):
        numberOfFeatures = len(id_from_tokenList)
        linesToWrite = []
        numberOfIdenticals = 0
        accumulatedActualProbability = 0
        accumulatedPredictedProbability = 0
        
        for executionIndex in range(numberOfParallelExecutions):
            
            ProbabilityOfActualsString = "Prob of actual: "
            ProbabilityOfPredictionsString= "Prob of pred: "
            predictionsString = "Predicted: " 
            actualsString = "Actual: " 
            
            for featureIndex in range(numberOfFeatures):

                prediction = TranslateToString(predictions[featureIndex][executionIndex])  
                actual     = TranslateToString(actuals[executionIndex][featureIndex])
                
                actualsString += (actual + ", ")

                if (actual == prediction):
                    predictionsString += " Identical "
                    numberOfIdenticals += 1
                else:
                    predictionsString += (prediction + ", ")
                    
                probabilityOfActual         = probabilityOfActuals[featureIndex][executionIndex]
                ProbabilityOfActualsString += "{:2.3f} ".format(probabilityOfActual)

                probabilityOfPrediction         = probabilityOfPredictions[featureIndex][executionIndex]
                ProbabilityOfPredictionsString += "{:2.3f} ".format(probabilityOfPrediction)    

                accumulatedActualProbability      += probabilityOfActual
                accumulatedPredictedProbability += probabilityOfPrediction

            lineToWrite = actualsString  + ProbabilityOfActualsString + "  " + predictionsString + ProbabilityOfPredictionsString + PrintfList[executionIndex]
            linesToWrite.append(lineToWrite)

        return linesToWrite, numberOfIdenticals, accumulatedActualProbability, accumulatedPredictedProbability




def ComposeLineListColored(numberOfParallelExecutions, nextPredictedTokens, nextTokens, PrintfList, probabilityOfNextTokens, probabilityPredictions):
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
     
    
def InferenceOutputThread(workerParameter, threadId, numberOfParallelExecutions, predictions, nextTokens, PrintfList, probabilityOfActuals, probabilityOfPredictions):
    with tf.device('/CPU:0'):
        id_from_tokenList = workerParameter
        linesToWrite, numberOfIdenticals, accumulatedActualProbability, accumulatedPredictedProbability = ComposeLineList(numberOfParallelExecutions, id_from_tokenList, predictions, nextTokens, PrintfList, probabilityOfActuals, probabilityOfPredictions)
        return linesToWrite, numberOfIdenticals, accumulatedActualProbability, accumulatedPredictedProbability


