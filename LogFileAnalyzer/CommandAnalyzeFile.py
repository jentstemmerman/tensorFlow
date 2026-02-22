import DefaultTrainingSettings
from Mask import Mask
import Model
import tensorflow as tf
import worker
import TokenParser
import Dataset

def AnalyzeFile(modelDirectory, userId, inputFile, inferenceDataSetFile, LogFileAnalyzedPath, signatureInputDirectory, similarityOutputDirectory, OriginalTokenFileDirectory, itemFile, temperature, progress_cb=None):
     
    success = False
    # tf.config.run_functions_eagerly(True)  
    # tf.data.experimental.enable_debug_mode()  
    physical_devices = tf.config.list_physical_devices("CPU")
    numberOfThreads = 2  # 1 for the main process
    print("Calling CreateWorkers numberOfThreads: ", numberOfThreads, "/ physical_devices: ", physical_devices)

    itemList                             = TokenParser.GetItemListToSearchFor(itemFile)
    id_from_tokenList, token_from_idList = Dataset.GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory=OriginalTokenFileDirectory, itemList=itemList) 
    modelFeatureList                     = Model.CreateModelFeatureListFromItemList(itemList, id_from_tokenList)
    maskValueList                        = Mask.GetMaskValueList(id_from_tokenList, itemList)

    TrainingsParameter = DefaultTrainingSettings.TrainingsParameter
    modelSource = {"Source": "FromFile", "ModelDirectory": modelDirectory, "Network": TrainingsParameter["Model"]["Network"]}
    model       = Model.GetModel(modelSource, maskValueList, modelFeatureList, TrainingsParameter)
    progress_cb(progress=1, status="Creating workers")   
    Workers = worker.CreateWorkers(numberOfThreads, id_from_tokenList)

    if Workers is None:
        print("Could not start workers. END")
    else:
        # Ensure we have a callable progress_cb to avoid conditional branches deeper
        if progress_cb is None:
            def progress_cb(**kwargs):
                return

        Model.AnalyzePrintfFile(
            model=model,
            userId=userId,
            progress_cb=progress_cb,
            inputFile=inputFile,
            inferenceDataSetFile=inferenceDataSetFile,
            LogFileAnalyzedPath=LogFileAnalyzedPath,
            signatureInputDirectory=signatureInputDirectory,
            similarityOutputDirectory=similarityOutputDirectory,
            itemFile=itemFile,
            temperature=temperature,
            seq_length=TrainingsParameter["Model"]["seq_length"],
            id_from_tokenList=id_from_tokenList,
            token_from_idList=token_from_idList,
            numberOfThreads=numberOfThreads,
            Workers=Workers,
            DataTypeForTensors=TrainingsParameter["DataTypeForTensors"],
        )

        success = True
        worker.KillWorkers(Workers)
    
        if progress_cb:
            try:
                progress_percent = int(100)
                progress_cb(progress=progress_percent, status="finished")
            except Exception as e:
                print(f"Progress callback failed: {e}")
    
    return success

