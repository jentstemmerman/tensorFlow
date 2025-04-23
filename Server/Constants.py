import os
from pathlib import Path


DATA_BASE_DIRECTORY = os.path.join("/" , "home", "jents", "AI", "data")


LOG_FILE_ANALYZER_DIRECTORY = os.path.join("/" , "home", "jents", "AI", "LogFileAnalyzer")
LOG_FILES_DIRECTORY = os.path.normpath("logFiles")
ITEM_FILES_DIRECTORY = os.path.normpath("ItemFiles")
ANALYZED_LOG_FILES_DIRECTORY = os.path.normpath("logFilesAnalyzed")
DATASET_ANALYZED_LOGFILES_DIRECTORY = os.path.normpath(os.path.join("cache", "Dataset_AnalyzedLogFiles"))
SIGNATURE_FILES_DIRECTORY = os.path.normpath("signatures")
MODEL_FILES_DIRECTORY = os.path.normpath("Models")
SETTINGS_DIRECTORY = os.path.normpath("settings")
OFFSET_FILES_DIRECTORY = os.path.normpath(os.path.join("cache", "offsets"))
ANNOTATION_FILES_DIRECTORY = os.path.normpath(os.path.join("cache", "annotations"))
USER_SETTINGS_FILE_NAME = "UserSettings.json"
TOKEN_FILE_DIRECTORY = "TokenFiles"

def GetLogFilesExtention():
    return ".log"

def GetAnnotationFilesExtention():
    return ".annotation"

def GetSignatureFilesExtention():
    return ".dat"

def GetAnnotationsDirectoryPath(logFilename, userId):
    userId = str(userId)

    baseDirectory = GetUserSpecificBaseDirectory("AnnotationType", userId)
    
    return baseDirectory

def GetLogFilePath(logFilename, userId):
    userId = str(userId)
    logFilename = os.path.basename(logFilename)
    baseDirectory = GetUserSpecificBaseDirectory("LogType", userId)

    log_file_path = os.path.join(baseDirectory, logFilename)
    return log_file_path

def GetLogFileAnalyzedPath(logFilename, userId):
    userId = str(userId)
    logFilenameWithoughExtention = Path(logFilename).stem
            
    baseDirectory = GetUserSpecificBaseDirectory("logFilesAnalyzed", userId)
    
    FileNameAnalyzedLogFile = os.path.join(baseDirectory, logFilenameWithoughExtention, "Analyzed.txt")

    return FileNameAnalyzedLogFile

def GetSignatureFilePath(signatureName, userId):
    userId = str(userId)
    signatureName = os.path.basename(signatureName)
    baseDirectory = GetUserSpecificBaseDirectory("SignatureType", userId)

    signature_file_path = os.path.join(baseDirectory, signatureName + GetSignatureFilesExtention())
    return signature_file_path

def GetMachineReadableSimularitiesFile(logFilename, userId, SignatureFile):
    userId = str(userId)
    
    logFilenameWithoughExtention = os.path.splitext(os.path.basename(logFilename))[0]
    SignatureName = os.path.splitext(os.path.basename(SignatureFile))[0]
       
    baseDirectory = GetUserSpecificBaseDirectory("logFilesAnalyzed", userId)
    
    FileNameSimularitiesWithSignature = os.path.join(baseDirectory, logFilenameWithoughExtention, "Simularities_" + SignatureName + GetSignatureFilesExtention())

    return FileNameSimularitiesWithSignature

def GetHumanReadableSimularitiesFile(logFilename, userId, SignatureFile):
    userId = str(userId)
    
    logFilenameWithoughExtention = os.path.splitext(os.path.basename(logFilename))[0]
    SignatureName = os.path.splitext(os.path.basename(SignatureFile))[0]
    
    baseDirectory = GetUserSpecificBaseDirectory("logFilesAnalyzed", userId)
    
    FileNameSimularitiesWithSignature = os.path.join(baseDirectory, logFilenameWithoughExtention, "Simularities_" + SignatureName + ".txt")

    return FileNameSimularitiesWithSignature

def GetLogFileMachineReadableVectorsPath(logFilename, userId):
    userId = str(userId)
    logFilenameWithoughExtention = os.path.splitext(os.path.basename(logFilename))[0]
            
    baseDirectory = GetUserSpecificBaseDirectory("logFilesAnalyzed", userId)
    
    FileNameMachineReadablePooledVectors_first              = os.path.join(baseDirectory, logFilenameWithoughExtention, "MachineReadablePooledVectors_first.dat")
    FileNameMachineReadablePooledVectors_last               = os.path.join(baseDirectory, logFilenameWithoughExtention, "MachineReadablePooledVectors_last.dat")
    FileNameMachineReadablePooledVectors_predicted          = os.path.join(baseDirectory, logFilenameWithoughExtention, "MachineReadablePooledVectors_predicted.dat")
    FileNameMachineReadablePooledVectors_all                = os.path.join(baseDirectory, logFilenameWithoughExtention, "MachineReadablePooledVectors_all.dat")
    FileNameMachineReadablePooledVectors_weighted_all       = os.path.join(baseDirectory, logFilenameWithoughExtention, "MachineReadablePooledVectors_weighted_all.dat")
    FileNameMachineReadablePooledVectors_attention_received = os.path.join(baseDirectory, logFilenameWithoughExtention, "MachineReadablePooledVectors_attention_received.dat")
    FileNameMachineReadablePooledVectors_all_attention_received = os.path.join(baseDirectory, logFilenameWithoughExtention, "MachineReadablePooledVectors_all_attention_received.dat")
          
    FileNameMachineReadablePooledVectorsDict = {"first": FileNameMachineReadablePooledVectors_first, 
                                                "last": FileNameMachineReadablePooledVectors_last, 
                                                "predicted": FileNameMachineReadablePooledVectors_predicted, 
                                                "all": FileNameMachineReadablePooledVectors_all,
                                                "all_weighted": FileNameMachineReadablePooledVectors_weighted_all,
                                                "attention_received": FileNameMachineReadablePooledVectors_attention_received,
                                                "all_attention_received": FileNameMachineReadablePooledVectors_all_attention_received}
    
    
    return FileNameMachineReadablePooledVectorsDict

def GetLogFileHumanReadableVectorsPath(logFilename, userId, temperature):
    userId = str(userId)
    logFilenameWithoughExtention = os.path.splitext(os.path.basename(logFilename))[0]
            
    baseDirectory = GetUserSpecificBaseDirectory("logFilesAnalyzed", userId)
    
    FileNameHumanReadablePooledVectors_first     = os.path.join(baseDirectory, logFilenameWithoughExtention, "HumanReadablePooledVectors_first.txt")
    FileNameHumanReadablePooledVectors_last      = os.path.join(baseDirectory, logFilenameWithoughExtention, "HumanReadablePooledVectors_last.txt")
    FileNameHumanReadablePooledVectors_predicted = os.path.join(baseDirectory, logFilenameWithoughExtention, "HumanReadablePooledVectors_predicted.txt")
    FileNameHumanReadablePooledVectors_all       = os.path.join(baseDirectory, logFilenameWithoughExtention, "HumanReadablePooledVectors_all.txt")
    FileNameHumanReadablePooledVectors_all_weighted  = os.path.join(baseDirectory, logFilenameWithoughExtention, "HumanReadablePooledVectors_all_weighted.txt")
    FileNameHumanReadablePooledVectors_attention_received = os.path.join(baseDirectory, logFilenameWithoughExtention, "HumanReadablePooledVectors_attention_received.txt")
    FFileNameHumanReadablePooledVectors_all_attention_received = os.path.join(baseDirectory, logFilenameWithoughExtention, "HumanReadablePooledVectors_all_attention_received.txt")


    FileNameHumanReadablePooledVectors = {"first": FileNameHumanReadablePooledVectors_first, 
                                          "last": FileNameHumanReadablePooledVectors_last, 
                                          "predicted": FileNameHumanReadablePooledVectors_predicted, 
                                          "all": FileNameHumanReadablePooledVectors_all,
                                          "all_weighted": FileNameHumanReadablePooledVectors_all_weighted,                                                
                                          "attention_received": FileNameHumanReadablePooledVectors_attention_received,
                                          "all_attention_received": FFileNameHumanReadablePooledVectors_all_attention_received}

    return FileNameHumanReadablePooledVectors

def GetInferenceDataSetPath(logFilename, userId, temperature):
    userId = str(userId)
    logFilenameWithoughExtention = os.path.splitext(os.path.basename(logFilename))[0]
            
    baseDirectory = GetUserSpecificBaseDirectory("inferenceDataSet", userId)
    
    fileName = os.path.join(baseDirectory, logFilenameWithoughExtention, "InferenceDataset")

    return fileName

def GetItemFilesPath():
    return GetNonUserSpecificBaseDirectory("ItemFiles")

def GetTokenFilesPath():
    return GetNonUserSpecificBaseDirectory("TokenFiles")

def GetOffsetsDirectoryPath(userId):
    userId = str(userId)
    baseDirectory = GetUserSpecificBaseDirectory("OffsetsType", userId)

    offset_file_path = baseDirectory
    return offset_file_path


def GetSettingsDirectoryPath(userId):
    userId = str(userId)
    baseDirectory = GetUserSpecificBaseDirectory("SettingsType", userId)

    settings_file_path = baseDirectory
    return settings_file_path


def GetModelsDirectoryPath():
    baseDirectory = GetNonUserSpecificBaseDirectory("ModelDirType")
    return baseDirectory

def GetSignatureDirectoryPath(userId):
    baseDirectory = GetUserSpecificBaseDirectory("SignatureType", userId)
    return baseDirectory

def GetNonUserSpecificBaseDirectory(directoryType):
    userSpecificDirectory = True
    
    if directoryType == "ModelDirType":
        userSpecificDirectory = False
        base_directory = MODEL_FILES_DIRECTORY
    elif directoryType == "ItemFiles":
        userSpecificDirectory = False
        base_directory = ITEM_FILES_DIRECTORY
    elif directoryType == "TokenFiles":
        userSpecificDirectory = False
        base_directory = TOKEN_FILE_DIRECTORY
    else:
        base_directory = None
        exit("Invalid directory type")

    assert not userSpecificDirectory, "Non User specific directory expected"

    base_directory = os.path.join(DATA_BASE_DIRECTORY, base_directory)
    return base_directory

def GetUserSpecificBaseDirectory(directoryType, userId):
    userSpecificDirectory = False
    if directoryType == "LogType":
        userSpecificDirectory = True
        base_directory = LOG_FILES_DIRECTORY
    elif (directoryType == "logFilesAnalyzed"): 
        userSpecificDirectory = True
        base_directory = ANALYZED_LOG_FILES_DIRECTORY
    elif (directoryType == "inferenceDataSet"): 
        userSpecificDirectory = True
        base_directory = DATASET_ANALYZED_LOGFILES_DIRECTORY
    elif directoryType == "AnnotationType":
        userSpecificDirectory = True
        base_directory = ANNOTATION_FILES_DIRECTORY
    elif directoryType == "SignatureType":
        userSpecificDirectory = True
        base_directory = SIGNATURE_FILES_DIRECTORY
    elif directoryType == "SettingsType":
        userSpecificDirectory = True
        base_directory = SETTINGS_DIRECTORY
    elif directoryType == "OffsetsType":
        userSpecificDirectory = True
        base_directory = OFFSET_FILES_DIRECTORY
    else:
        base_directory = None
        exit("Invalid directory type")

    assert userSpecificDirectory, "User specific directory expected"
    
    base_directory = os.path.join("User_" + str(userId), base_directory)
    base_directory = os.path.join("UserData", base_directory)
    base_directory = os.path.join(DATA_BASE_DIRECTORY, base_directory)

    return base_directory