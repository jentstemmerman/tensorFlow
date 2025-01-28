import os

LOG_FILES_DIRECTORY = os.path.normpath("./logFiles")
SIGNATURE_FILES_DIRECTORY = os.path.normpath("./signatures")
MODEL_FILES_DIRECTORY = os.path.normpath("./models")
SETTINGS_DIRECTORY = os.path.normpath("./settings")
OFFSET_FILES_DIRECTORY = os.path.normpath("./cache/offsets")
ANNOTATION_FILES_DIRECTORY = os.path.normpath("./cache/annotations")
USER_SETTINGS_FILE_NAME = "UserSettings.json"


def GetAnnotationsDirectoryPath(logFilename, userId):
    userId = str(userId)
    logFilename = os.path.basename(logFilename)
    baseDirectory = GetBaseDirectory("AnnotationType", userId)

    annotations_file_path = baseDirectory
    return annotations_file_path


def GetLogFilePath(logFilename, userId):
    userId = str(userId)
    logFilename = os.path.basename(logFilename)
    baseDirectory = GetBaseDirectory("LogType", userId)

    log_file_path = os.path.join(baseDirectory, logFilename)
    return log_file_path


def GetOffsetsDirectoryPath(logFilename, userId):
    userId = str(userId)
    logFilename = os.path.basename(logFilename)
    baseDirectory = GetBaseDirectory("OffsetsType", userId)

    offset_file_path = baseDirectory
    return offset_file_path


def GetSettingsDirectoryPath(userId):
    userId = str(userId)
    baseDirectory = GetBaseDirectory("SettingsType", userId)

    settings_file_path = baseDirectory
    return settings_file_path


def GetBaseDirectory(directoryType, userId):
    data_directory = "user_data"
    if directoryType == "LogType":
        base_directory = LOG_FILES_DIRECTORY
    elif directoryType == "AnnotationType":
        base_directory = ANNOTATION_FILES_DIRECTORY
    elif directoryType == "SignatureType":
        base_directory = SIGNATURE_FILES_DIRECTORY
    elif directoryType == "ModelDirType":
        base_directory = MODEL_FILES_DIRECTORY
    elif directoryType == "SettingsType":
        base_directory = SETTINGS_DIRECTORY
    elif directoryType == "OffsetsType":
        base_directory = OFFSET_FILES_DIRECTORY
    else:
        base_directory = None
        exit("Invalid directory type")

    base_directory = os.path.join(data_directory, base_directory)
    base_directory = os.path.join("User_" + str(userId), base_directory)
    base_directory = os.path.join("../data", base_directory)

    return base_directory
