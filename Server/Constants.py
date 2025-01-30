import os

LOG_FILES_DIRECTORY = os.path.normpath("logFiles")
ITEM_FILES_DIRECTORY = os.path.normpath("ItemFiles")
ANALYZED_LOG_FILES_DIRECTORY = os.path.normpath("logFilesAnalyzed")
SIGNATURE_FILES_DIRECTORY = os.path.normpath("signatures")
MODEL_FILES_DIRECTORY = os.path.normpath("Models")
SETTINGS_DIRECTORY = os.path.normpath("settings")
OFFSET_FILES_DIRECTORY = os.path.normpath(os.path.join("cache", "offsets"))
ANNOTATION_FILES_DIRECTORY = os.path.normpath(os.path.join("cache", "annotations"))
USER_SETTINGS_FILE_NAME = "UserSettings.json"
TOKEN_FILE_DIRECTORY = "TokenFiles"

def GetAnnotationsDirectoryPath(logFilename, userId):
    userId = str(userId)
    logFilename = os.path.basename(logFilename)
    baseDirectory = GetUserSpecificBaseDirectory("AnnotationType", userId)

    annotations_file_path = baseDirectory
    return annotations_file_path


def GetLogFilePath(logFilename, userId):
    userId = str(userId)
    logFilename = os.path.basename(logFilename)
    baseDirectory = GetUserSpecificBaseDirectory("LogType", userId)

    log_file_path = os.path.join(baseDirectory, logFilename)
    return log_file_path

def GetLogFileAnalyzedPath(logFilename, userId):
    userId = str(userId)
    logFilename = os.path.basename(logFilename)
    baseDirectory = GetUserSpecificBaseDirectory("logFilesAnalyzed", userId)

    return os.path.join(baseDirectory, logFilename + ".analysed")

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

    base_directory = os.path.join(os.path.join("..", "data"), base_directory)
    return base_directory

def GetUserSpecificBaseDirectory(directoryType, userId):
    userSpecificDirectory = False
    if directoryType == "LogType":
        userSpecificDirectory = True
        base_directory = LOG_FILES_DIRECTORY 
    elif (directoryType == "logFilesAnalyzed"): 
        userSpecificDirectory = True
        base_directory = ANALYZED_LOG_FILES_DIRECTORY
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
    base_directory = os.path.join(os.path.join("..", "data"), base_directory)

    return base_directory