import TokenParser
import tensorflow as tf
import numpy as np
from numbers import Number
import Dataset

def extend_string_with_spaces(s, target_length):
    if len(s) >= target_length:
        return s  # String is already at or beyond the target length

    spaces_to_add = target_length - len(s)
    extended_string = s + ' ' * spaces_to_add
    return extended_string

def TranslateToString(value):
    if isinstance(value, (Number, np.number)):
        value = "{:5.2f} ".format(value)
    elif tf.is_tensor(value):
        value = value.numpy()[0].decode('ascii')
    else:
        value = str(value)
    return value



def ComposeHumanReadablePooledLogitsVectorsList(numberOfParallelExecutions, pooledLogitsVector):
    with tf.device('/CPU:0'):
        linesToWrite = []
        for executionIndex in range(numberOfParallelExecutions):
            arrayString = np.array2string(pooledLogitsVector['vector'][executionIndex].numpy(), max_line_width=80, threshold=1000)
            line = TranslateToString(pooledLogitsVector['similarity'][executionIndex].numpy()) + "   " + arrayString + "\n"
            linesToWrite.append(line)

        return linesToWrite

def ComposeDensePooledLogitsVectorsList(numberOfParallelExecutions, pooledLogitsVector):
    with tf.device('/CPU:0'):
        vectorsToWrite = []
        for executionIndex in range(numberOfParallelExecutions):
            vector = pooledLogitsVector['vector'][executionIndex].numpy()
            vectorsToWrite.append(vector)
        return vectorsToWrite

def ComposeLineList(numberOfParallelExecutions, 
                    id_from_tokenList, 
                    predictions, 
                    actuals, 
                    PrintfList, 
                    probabilityOfActuals, 
                    crossEntropyOfActuals, 
                    probabilityOfPredictions, 
                    pooledLogitsVector):
    
    with tf.device('/CPU:0'):
        string_length = 17
        number_string_length = 6
        numberOfFeatures = len(id_from_tokenList)
        linesToWrite = []
        localPrintfList = []
        numberOfIdenticals = 0
        accumulatedActualProbability = 0
        accumulatedPredictedProbability = 0
        accumulatedPredictionAccuracy = 0
        accumulatedCrossEntropyOfActual = 0

        IdenticalString =  " (ID)"
        for executionIndex in range(numberOfParallelExecutions):
            
            ProbabilityOfActualsString = "Prob of actual: "
            ProbabilityOfPredictionsString= "Prob of pred: "
            predictionsString = ". Predicted: " 
            actualsString = "Actual: " 
            crossEntropyOfActualString = "ActualCrEntr: "
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

                crossEntropyOfActual        = crossEntropyOfActuals[featureIndex][executionIndex]
                crossEntropyOfActualString += "{:2.3f} ".format(crossEntropyOfActual)
                crossEntropyOfActualString  = extend_string_with_spaces(crossEntropyOfActualString, number_string_length)

                probabilityOfPrediction         = probabilityOfPredictions[featureIndex][executionIndex]
                ProbabilityOfPredictionsString += "{:2.3f} ".format(probabilityOfPrediction)    
                ProbabilityOfPredictionsString  = extend_string_with_spaces(ProbabilityOfPredictionsString, number_string_length)

                AccuracyOfPrediction += 1-(probabilityOfPrediction-probabilityOfActual)
                
                
                accumulatedPredictionAccuracy   += AccuracyOfPrediction
                accumulatedActualProbability    += probabilityOfActual
                accumulatedPredictedProbability += probabilityOfPrediction
                accumulatedCrossEntropyOfActual += crossEntropyOfActual

            AccuracyOfPrediction = AccuracyOfPrediction / numberOfFeatures
            AccuracyOfPredictionString  = "HIDDEN" + str(int(AccuracyOfPrediction * 100))    
            AccuracyOfPredictionString  = extend_string_with_spaces(AccuracyOfPredictionString, 10)

            SimilarityString = " Simul: " + TranslateToString(pooledLogitsVector['similarity'][executionIndex].numpy()) + "  "
            
            #print(f"PrintfList[{executionIndex}], numberOfParallelExecutions = {numberOfParallelExecutions} len(PrintfList) = {len(PrintfList)}")
            
            if (executionIndex >= len(PrintfList)):
                print(f"Error")
            
            
            printf = PrintfList[executionIndex]
            lineToWrite = AccuracyOfPredictionString + SimilarityString + actualsString + predictionsString + ProbabilityOfActualsString + crossEntropyOfActualString + "  "  + ProbabilityOfPredictionsString + printf
 
            localPrintfList.append(printf)
            linesToWrite.append(lineToWrite)

        return linesToWrite, localPrintfList, numberOfIdenticals, accumulatedActualProbability, accumulatedPredictedProbability




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
     
    
def InferenceOutputThread(workerParameter, 
                          threadId, 
                          numberOfParallelExecutions, 
                          predictions, 
                          nextTokens, 
                          PrintfList, 
                          probabilityOfActuals, 
                          crossEntropyOfActuals, 
                          probabilityOfPredictions, 
                          pooledLogitsVector):
    
    with tf.device('/CPU:0'):
        id_from_tokenList = workerParameter[0]
        token_from_idList = workerParameter[1]
        
        linesToWrite, printfList, numberOfIdenticals, accumulatedActualProbability, accumulatedPredictedProbability = ComposeLineList(numberOfParallelExecutions=numberOfParallelExecutions, 
                                                                                                                                      id_from_tokenList=id_from_tokenList, 
                                                                                                                                      predictions=predictions, 
                                                                                                                                      actuals=nextTokens, 
                                                                                                                                      PrintfList=PrintfList, 
                                                                                                                                      probabilityOfActuals=probabilityOfActuals, 
                                                                                                                                      crossEntropyOfActuals=crossEntropyOfActuals, 
                                                                                                                                      probabilityOfPredictions=probabilityOfPredictions, 
                                                                                                                                      pooledLogitsVector=pooledLogitsVector)
        
        pooledHumanReadableLogitsVectorlinesToWrite = ComposeHumanReadablePooledLogitsVectorsList(numberOfParallelExecutions, pooledLogitsVector)
        pooledMachineReadableLogitsVectorlinesToWrite = ComposeDensePooledLogitsVectorsList(numberOfParallelExecutions, pooledLogitsVector)
        
        return linesToWrite, printfList, numberOfIdenticals, accumulatedActualProbability, accumulatedPredictedProbability, pooledHumanReadableLogitsVectorlinesToWrite, pooledMachineReadableLogitsVectorlinesToWrite


