import sys
sys.path.append('../LogFileAnalyzer')
import hashlib
import os
import OffsetArrayManager
import AnnotationManager
import time
import Constants
import LogFileIterator    # type: ignore

fileHandle = None
file = "./Server/large-file.txt"
offsetArrayManager = None
annotationManager = None


def read_file_lines(file_path, userId, lineOffset, numberOfLines):
    startOffset = get_line_offset(file_path, userId, lineOffset)
    endOffset = get_line_offset(file_path, userId, lineOffset + numberOfLines - 1)
    numberOfBytesToRead = endOffset - startOffset

    chunk = read_file_chunk(file_path, startOffset, numberOfBytesToRead)

    return chunk


def open_file_once(file_path):
    """Open the file once and keep the handle in global fileHandle."""
    global fileHandle
    if fileHandle is None or fileHandle.name != file_path:
        if fileHandle is not None:
            fileHandle.close()
        try:
            fileHandle = open(file_path, "rb")
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
        except IOError as e:
            print(f"IO error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


def read_file_chunk(file_path, offset, size):
    global fileHandle
    chunk = None
    # Ensure the file is opened once
    open_file_once(file_path)

    # If fileHandle is valid, perform the read
    if fileHandle:
        try:
            fileHandle.seek(offset)
            chunk = fileHandle.read(size)
        except IOError as e:
            print(f"IO error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    return chunk


def get_line_offset(file_path, userId, lineNumber):
    lineOffsetsArray = GetOffsetsArray(file_path, userId)

    offset = lineOffsetsArray[lineNumber]

    return offset


def GetOffsetsManager(userId):
    global offsetArrayManager
    # for multiple users, we need to have a dictionary of offsetArrayManagers
    if offsetArrayManager is None:
        offsetArrayManager = OffsetArrayManager.OffsetArrayManager(
            max_cache_size=3, userId=userId
        )

    return offsetArrayManager


def GetAnnotationManager(userId):
    global annotationManager

    # for multiple users, we need to have a dictionary of annotationManagers
    if annotationManager is None:
        annotationManager = AnnotationManager.AnnotationCacheManager(
            max_cache_size=3, userId=userId
        )

    return annotationManager


def GetNumberOfLinesInFile(file_path, userId):
    offsetArrayManager = GetOffsetsManager(userId)
    _, NumberOfLines = offsetArrayManager.get_offsets_array(file_path)

    return NumberOfLines


def GetOffsetsArray(file_path, userId):
    offsetArrayManager = GetOffsetsManager(userId)

    offsetsArray, _ = offsetArrayManager.get_offsets_array(file_path)

    return offsetsArray


def GetAnnotations(file_path, userId, startLine, numberOfLines, annotation_id):
    annotationManager = GetAnnotationManager(userId)

    annotations, annotationHeader = annotationManager.get_annotations(
        file_path, startLine, numberOfLines, annotation_id
    )

    return annotations, annotationHeader


def CheckAnnotations(file_path, userId, annotation_id):
    annotationManager = GetAnnotationManager(userId)

    annotationsAvailable = annotationManager.check_annotation_file_exists(
        file_path, annotation_id
    )

    return annotationsAvailable


def select_input_file(file):
    # Ensure file exists
    file_path = os.path.abspath(file)
    file_size = 0

    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        file_path = os.path.abspath(file)

    return (
        file_path,
        file_size,
    )
   
        
def PostProcessLogFiles(fileName, userId, DataTypeForTensors, similaritythreshold):
    import LogFileIterator    # type: ignore
    import SignatureMatching  # type: ignore     
    
    try:
        # Your asynchronous processing logic here
        print(f"async_preprocess_log_files user {userId}")
        
        logfileBaseDirectory = Constants.GetUserSpecificBaseDirectory("LogType", userId)
        signatureDirectory = Constants.GetUserSpecificBaseDirectory("SignatureType", userId)
        
        logFileIterator = LogFileIterator.FileIterator(logfileBaseDirectory,  Constants.GetLogFilesExtention())
                
        logFileName      = logFileIterator.get_first_tokenlogfile()
        
        while (logFileName is not None):
               
            logFileName_name = os.path.splitext(os.path.basename(logFileName))[0]
            
            print(f"Start Creating SimularitiesFiles of logfile {logFileName_name}")
       
            SignatureMatching.CreateSimularitiesFilesOptimized(logFileName, signatureDirectory, userId, DataTypeForTensors)                                                 
            
            print(f"Finished Creating SimularitiesFile of logfile {logFileName_name}")
                
            logFileName = logFileIterator.get_next_tokenlogfile()
                
        logFileName      = logFileIterator.get_first_tokenlogfile()
        
        signatureIterator = LogFileIterator.FileIterator(signatureDirectory, Constants.GetSignatureFilesExtention())
        signature_file   = signatureIterator.get_first_tokenlogfile()
        
        while (logFileName is not None):
            while (signature_file is not None):                
                signature_name = os.path.splitext(os.path.basename(signature_file))[0]
                annotation_id = signature_name
                
                annotationsOutputFile = AnnotationManager.AnnotationCacheManager.get_annotation_file_path(logFileName, annotation_id, userId)
                        
                simularitiesMachineReadableFileName = Constants.GetMachineReadableSimularitiesFile(logFileName, userId, signature_file)
                
                SignatureMatching.CreateAnnotationListFromSimularitiesFile(simularitiesMachineReadableFileName, annotationsOutputFile, DataTypeForTensors, annotationId = annotation_id, threshold = similaritythreshold, comment = signature_name)

                print(f"Finished Creating AnnotationList of logfile {logFileName_name} for signature{signature_name}")
                
                signature_file = signatureIterator.get_next_tokenlogfile()

            logFileName = logFileIterator.get_next_tokenlogfile()
        
        print(f"Finished processing log files for user {userId}")
    except Exception as e:
        print(f"Error processing log files for user {userId}: {e}")
        
def CreateSimularityFiles(logFileName, userId, DataTypeForTensors, similaritythreshold):
    import SignatureMatching  # type: ignore
    logfileBaseDirectory = Constants.GetUserSpecificBaseDirectory("LogType", userId)
    logFileName = os.path.join(logfileBaseDirectory, logFileName)
    try:
        print(f"CreateSimularityFiles {logFileName} user {userId}, similaritythreshold {similaritythreshold}")
        signatureDirectory = Constants.GetUserSpecificBaseDirectory("SignatureType", userId)
        SignatureMatching.CreateSimularitiesFilesOptimized(logFileName, signatureDirectory, userId, DataTypeForTensors)     
    except Exception as e:
        print(f"Error CreateSimularityFiles {logFileName} {userId}: {e}")     


def CreateAnnotationListFromSimularityFile(logFileName, signature_file, userId, DataTypeForTensors, similaritythreshold):
    signature_name = os.path.splitext(os.path.basename(signature_file))[0]
    annotation_id = signature_name
    annotationsOutputFile = AnnotationManager.AnnotationCacheManager.get_annotation_file_path(logFileName, annotation_id, userId)
    simularitiesMachineReadableFileName = Constants.GetMachineReadableSimularitiesFile(logFileName, userId, signature_file)
    
    import SignatureMatching  # type: ignore   
    
    SignatureMatching.CreateAnnotationListFromSimularitiesFile(simularitiesMachineReadableFileName, annotationsOutputFile, DataTypeForTensors, annotationId = annotation_id, threshold = similaritythreshold, comment = signature_name)

    
def PostProcessLogFile(logFileName, userId, DataTypeForTensors, similaritythreshold):
    try:

        print(f"PostProcessLogFile {logFileName} user {userId}")
        
        CreateSimularityFiles(logFileName, userId, DataTypeForTensors, similaritythreshold)
            
        signatureDirectory = Constants.GetUserSpecificBaseDirectory("SignatureType", userId)
        signatureIterator = LogFileIterator.FileIterator(signatureDirectory, Constants.GetSignatureFilesExtention())
        
        signature_file   = signatureIterator.get_first_tokenlogfile()
       

        while (signature_file is not None):                
            CreateAnnotationListFromSimularityFile(logFileName, signature_file, userId, DataTypeForTensors, similaritythreshold)
            signature_file = signatureIterator.get_next_tokenlogfile()

        
        print(f"Finished processing log files for user {userId}")
    except Exception as e:
        print(f"Error processing log files for user {userId}: {e}")