import TokenParser
import tensorflow as tf
import numpy as np
from numbers import Number
import Dataset
import time

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

def ComposeHumanReadablePooledLogitsVectorsList(numberOfParallelExecutions, pooledLogitsVector_hidden_state_last):
    with tf.device('/CPU:0'):
        linesToWrite = []
        for executionIndex in range(numberOfParallelExecutions):
            arrayString = np.array2string(
                    pooledLogitsVector_hidden_state_last[executionIndex], 
                    max_line_width=800, 
                    threshold=1000, 
                    separator=' ',  # Ensure consistent separation
                    formatter={'float_kind': lambda x: f"{x:8.7f}"}  # Adjust width (e.g., 6 chars wide, 2 decimals)
                )
            
            line = arrayString + "\n"
            linesToWrite.append(line)

        return linesToWrite

def ComposeHumanReadablePooledLogitsAllVectorsList(numberOfParallelExecutions, numberOfAnalyzedLines, sequenceLength, pooledLogitsVector_hidden_state_last):
    with tf.device('/CPU:0'):
        linesToWrite = []
        index = 0
        
        analyzedLine = numberOfAnalyzedLines + 1 # +1 because we predict the next token
        
        for lineIndex in range(numberOfParallelExecutions):
            linesToWrite.append(f"Lines [{analyzedLine} - {analyzedLine + sequenceLength - 1}]:\n") 
            for sequenceIndex in range(sequenceLength):
                arrayString = np.array2string(
                    pooledLogitsVector_hidden_state_last[index], 
                    max_line_width=800, 
                    threshold=1000, 
                    separator=' ',  # Ensure consistent separation
                    formatter={'float_kind': lambda x: f"{x:8.3f}"}  # Adjust width (e.g., 6 chars wide, 2 decimals)
                )

                linesToWrite.append(f"{analyzedLine + sequenceIndex:04d}: {arrayString} \n")
                index += 1
            
            analyzedLine += 1
      
        return linesToWrite
    
def ComposeLineList(numberOfParallelExecutions, 
                    id_from_tokenList, 
                    token_from_idList,
                    predictions, 
                    actualNextIdList, 
                    PrintfList, 
                    probabilityOfActuals, 
                    crossEntropyOfActuals, 
                    probabilityOfPredictions):
    
    with tf.device('/CPU:0'):
        string_length = 17
        number_string_length = 6
        numberOfFeatures = len(id_from_tokenList)
        linesToWrite = []
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
                
                if (token_from_idList[featureIndex] is None):
                    actual = TranslateToString(actualNextIdList[executionIndex][featureIndex])
                else:
                    actual = TranslateToString(token_from_idList[featureIndex](actualNextIdList[executionIndex][featureIndex])) # TranslateToString("DISABLED") #
                    # actual = TranslateToString("DISABLED")  # TranslateToString("DISABLED") #    
                
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

            #print(f"PrintfList[{executionIndex}], numberOfParallelExecutions = {numberOfParallelExecutions} len(PrintfList) = {len(PrintfList)}")
            
            if (executionIndex >= len(PrintfList)):
                print(f"Warning (executionIndex >= len(PrintfList))")
                printf = "No printf left"
            else:
                printf = PrintfList[executionIndex]
                
            lineToWrite = AccuracyOfPredictionString + actualsString + predictionsString + ProbabilityOfActualsString + crossEntropyOfActualString + "  "  + ProbabilityOfPredictionsString + printf
 
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
     
    
def InferenceOutputThread(workerParameter, 
                          threadId, 
                          numberOfParallelExecutions, 
                          predictions, 
                          actualNextIdList, 
                          PrintfList, 
                          probabilityOfActuals, 
                          crossEntropyOfActuals, 
                          probabilityOfPredictions, 
                          pooledLogitsVector_hidden_state_all,
                          attention_receivedVectorDataList,
                          pooledLogitsVector_hidden_state_last,
                          sequenceLength,
                          numberOfAnalyzedLines,
                          onlyLogLogitsVectors=False):
    
    with tf.device('/CPU:0'):
       
        if onlyLogLogitsVectors:
            HumanReadableTextOutput = None
            pooledHumanReadableLogitsLastVectorlinesToWrite = None
            pooledHumanReadableLogitsAllVectorlinesToWrite = None
            attention_receivedVectorlinesToWrite = None
            all_attention_receivedVectorlinesToWrite = None
        else:
            id_from_tokenList = workerParameter[0]
            token_from_idList = workerParameter[1]
            
            HumanReadableTextOutput = ComposeLineList(  numberOfParallelExecutions=numberOfParallelExecutions, 
                                                        id_from_tokenList=id_from_tokenList, 
                                                        token_from_idList=token_from_idList,
                                                        predictions=predictions, 
                                                        actualNextIdList=actualNextIdList, 
                                                        PrintfList=PrintfList, 
                                                        probabilityOfActuals=probabilityOfActuals, 
                                                        crossEntropyOfActuals=crossEntropyOfActuals, 
                                                        probabilityOfPredictions=probabilityOfPredictions)
            
            pooledHumanReadableLogitsLastVectorlinesToWrite = ComposeHumanReadablePooledLogitsVectorsList(numberOfParallelExecutions, pooledLogitsVector_hidden_state_last)
            pooledHumanReadableLogitsAllVectorlinesToWrite  = ComposeHumanReadablePooledLogitsAllVectorsList(numberOfParallelExecutions, numberOfAnalyzedLines, sequenceLength, pooledLogitsVector_hidden_state_all)
            attention_receivedVectorlinesToWrite            = ComposeHumanReadablePooledLogitsVectorsList(numberOfParallelExecutions, attention_receivedVectorDataList)
            all_attention_receivedVectorlinesToWrite        =  None
            
            
        return HumanReadableTextOutput, pooledHumanReadableLogitsLastVectorlinesToWrite, pooledHumanReadableLogitsAllVectorlinesToWrite, attention_receivedVectorlinesToWrite, all_attention_receivedVectorlinesToWrite


