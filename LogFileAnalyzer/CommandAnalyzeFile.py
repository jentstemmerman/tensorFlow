import DefaultTrainingSettings
import Model
import tensorflow as tf
import worker
import TokenParser
import Dataset
import SignatureMatching

def AnalyzeFile(modelDirectory, userId, inputFile, inferenceDataSetFile, LogFileAnalyzedPath, signatureInputDirectory, similarityOutputDirectory, FileNamesMachineReadablePooledVectors, FileNamesHumanReadablePooledVectors, OriginalTokenFileDirectory, itemFile, temperature):
     
    success = False
    physical_devices = tf.config.list_physical_devices("CPU")
    numberOfThreads = 2  # 1 for the main process
    print("Calling CreateWorkers numberOfThreads: ", numberOfThreads, "/ physical_devices: ", physical_devices)

    itemList                             = TokenParser.GetItemListToSearchFor(itemFile)
    id_from_tokenList, token_from_idList = Dataset.GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory=OriginalTokenFileDirectory, itemList=itemList) 
    modelFeatureList                     = Model.CreateModelFeatureListFromItemList(itemList, id_from_tokenList)
    maskValueList                        = Model.GetMaskValueList(id_from_tokenList)

    TrainingsParameter = DefaultTrainingSettings.TrainingsParameter
    modelSource = {"Source": "FromFile", "ModelDirectory": modelDirectory, "Network": TrainingsParameter["Model"]["Network"]}
    model       = Model.GetModel(modelSource, maskValueList, modelFeatureList, TrainingsParameter)
        
#    numberOfSignatureFiles1, numberOfPooledVectorsList1, signatureNames1, signatureTensors1, signatureFiles1 = SignatureMatching.GetSignaturesFromDirectoryAndRecalculateDirectlyFromLogFile(model=model, userId=userId, itemList=itemList, id_from_tokenList=id_from_tokenList, token_from_idList=token_from_idList, temperature=temperature, window_size=TrainingsParameter["Model"]["seq_length"])
        
#    numberOfSignatureFiles2, numberOfPooledVectorsList2, signatureNames2, signatureTensors2, signatureFiles2 = SignatureMatching.GetSignaturesFromDirectory(userId, itemList)       
        
  
        
        
    Workers = worker.CreateWorkers(numberOfThreads, id_from_tokenList)

    if (Workers == None):
        print("Could not start workers. END")
    else:
        Model.AnalyzePrintfFile(    model=model, 
                                    userId=userId,
                                    inputFile=inputFile, 
                                    inferenceDataSetFile=inferenceDataSetFile,
                                    LogFileAnalyzedPath=LogFileAnalyzedPath,
                                    signatureInputDirectory=signatureInputDirectory,
                                    similarityOutputDirectory=similarityOutputDirectory,
                                    FileNamesMachineReadablePooledVectors=FileNamesMachineReadablePooledVectors, 
                                    FileNamesHumanReadablePooledVectors=FileNamesHumanReadablePooledVectors,
                                    itemFile=itemFile, 
                                    temperature=temperature, 
                                    seq_length=TrainingsParameter["Model"]["seq_length"], 
                                    id_from_tokenList = id_from_tokenList,
                                    token_from_idList=token_from_idList,
                                    numberOfThreads=numberOfThreads, 
                                    Workers=Workers,
                                    DataTypeForTensors=TrainingsParameter["DataTypeForTensors"] )

        success = True
        worker.KillWorkers(Workers)
    
    return success

