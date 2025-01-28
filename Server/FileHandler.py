import hashlib
import os
import OffsetArrayManager
import AnnotationManager

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

    annotations = annotationManager.get_annotations(
        file_path, startLine, numberOfLines, annotation_id
    )

    return annotations


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
