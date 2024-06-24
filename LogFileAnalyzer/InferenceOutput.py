import multiprocessing
import TokenParser
import time
import tensorflow as tf
import numpy as np
from numbers import Number

def extend_string_with_spaces(s, target_length):
    if len(s) >= target_length:
        return s  # String is already at or beyond the target length

    spaces_to_add = target_length - len(s)
    extended_string = s + ' ' * spaces_to_add
    return extended_string

def TranslateToString(value):
    if isinstance(value, (Number, np.number)):
        value = "{:5.2f} ".format(value)  
    else:
        value = value
    return value

import Dataset

def ComposeLineList(numberOfParallelExecutions, id_from_tokenList, predictions, actuals, PrintfList, probabilityOfActuals, probabilityOfPredictions):
    with tf.device('/CPU:0'):
        string_length = 17
        number_string_length = 6
        numberOfFeatures = len(id_from_tokenList)
        linesToWrite = []
        numberOfIdenticals = 0
        accumulatedActualProbability = 0
        accumulatedPredictedProbability = 0
        accumulatedPredictionAccuracy = 0

        IdenticalString =  " (ID)"
        for executionIndex in range(numberOfParallelExecutions):
            
            ProbabilityOfActualsString = "Prob of actual: "
            ProbabilityOfPredictionsString= "Prob of pred: "
            predictionsString = ". Predicted: " 
            actualsString = "Actual: " 
            AccuracyOfPrediction = 0

            for featureIndex in range(numberOfFeatures):

                prediction = TranslateToString(predictions[featureIndex][executionIndex])  
                actual     = TranslateToString(actuals[executionIndex][featureIndex])
                
                if (actual == prediction) and (actual != Dataset.Universal_Mask_Token):
                    prediction = prediction + IdenticalString
                    prediction = extend_string_with_spaces(prediction, string_length)
                    predictionsString  += prediction
                    numberOfIdenticals += 1
                else:
                    prediction = extend_string_with_spaces(prediction, string_length)
                    predictionsString  += prediction

                actual = extend_string_with_spaces(actual, string_length)                    

                
                actualsString += actual
                if (featureIndex < (numberOfFeatures-1)):
                    actualsString     += ", "
                    predictionsString += ", "
                    
                probabilityOfActual         = probabilityOfActuals[featureIndex][executionIndex]
                ProbabilityOfActualsString += "{:2.3f} ".format(probabilityOfActual)
                ProbabilityOfActualsString  = extend_string_with_spaces(ProbabilityOfActualsString, number_string_length)

                probabilityOfPrediction         = probabilityOfPredictions[featureIndex][executionIndex]
                ProbabilityOfPredictionsString += "{:2.3f} ".format(probabilityOfPrediction)    
                ProbabilityOfPredictionsString  = extend_string_with_spaces(ProbabilityOfPredictionsString, number_string_length)

                AccuracyOfPrediction += 1-(probabilityOfPrediction-probabilityOfActual)

                accumulatedPredictionAccuracy   += AccuracyOfPrediction
                accumulatedActualProbability    += probabilityOfActual
                accumulatedPredictedProbability += probabilityOfPrediction

            AccuracyOfPrediction = AccuracyOfPrediction / numberOfFeatures
            AccuracyOfPredictionString  = "HIDDEN" + str(int(AccuracyOfPrediction * 100))    
            AccuracyOfPredictionString  = extend_string_with_spaces(AccuracyOfPredictionString, 10)
            
            lineToWrite = AccuracyOfPredictionString + actualsString + predictionsString + ProbabilityOfActualsString + "  "  + ProbabilityOfPredictionsString + PrintfList[executionIndex]
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


