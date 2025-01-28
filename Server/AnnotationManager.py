import os
import hashlib
import logging
from collections import OrderedDict
from threading import Lock
import numpy as np
import LRUCache
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnnotationCacheManager:
    def __init__(self, max_cache_size=100, userId=None):
        self.annotation_cache = LRUCache.LRUCache(max_cache_size)
        self.cache_lock = Lock()

        self.userId = userId

    """
    Read the annotation file into a Python list.

    Parameters:
        annotation_file_path (str): The path to the annotation file.

    Returns:
        list: A list of annotations, each represented as a dictionary.
    """

    def read_annotation_file(self, file_path: str, annotation_id: int) -> list:
        annotation_file_path = self.get_annotation_file_path(file_path, annotation_id)

        annotations = []

        try:
            with open(annotation_file_path, "rb") as f:
                annotations = pickle.load(f)

        except (IOError, OSError) as e:
            logger.error(f"An error occurred while reading the annotation file: {e}")
        return annotations

    """
    Write annotations to a file.

    Parameters:
        file_path (str): The path to the source file.
        annotations (list): The annotations to write, each represented as a dictionary.
    """

    def write_annotation_file(
        self, file_path: str, annotations: list, annotation_id: int
    ):
        fileName = os.path.basename(file_path)
        annotation_file_path = self.get_annotation_file_path(fileName, annotation_id)

        try:
            os.makedirs(os.path.dirname(annotation_file_path), exist_ok=True)
            with open(annotation_file_path, "wb") as f:
                pickle.dump(annotations, f)
        except (IOError, OSError) as e:
            logger.error(f"An error occurred while writing to file: {e}")

    def check_annotation_file_exists(self, file_path: str, annotation_id: int) -> bool:
        annotation_file_path = self.get_annotation_file_path(file_path, annotation_id)
        return os.path.exists(annotation_file_path)

    """
    Retrieve annotations for a given file within a specified range of lines.

    Parameters:
        file_path (str): The path to the source file.
        startLine (int): The starting line number for filtering annotations.
        numberOfLines (int): The number of lines to include in the range.
        annotation_id (int): The Id of the annotation
    Returns:
        list: A list of annotations within the specified range.
    """

    def get_annotations(
        self, file_path: str, startLine: int, numberOfLines: int, annotation_id: int
    ) -> list:
        annotation_file_path = self.get_annotation_file_path(file_path, annotation_id)

        # print("[AnnotationManager] get_annotations ", startLine, " numberOfLines: ", numberOfLines)

        with self.cache_lock:  # Ensures thread safety
            cached_file_path = self.annotation_cache.get(file_path)
            cached_hash = self.annotation_cache.get(f"{file_path}_hash")

            annotation_file_exists = os.path.exists(annotation_file_path)
            annotation_file_hash = self.generate_file_hash(annotation_file_path)
            annotation_file_hash_match = annotation_file_hash == cached_hash

            if not annotation_file_exists:
                return None

            if (cached_file_path is None) or (not annotation_file_hash_match):
                # Update cache with new annotations and file hash
                annotations = self.read_annotation_file(
                    file_path=file_path, annotation_id=annotation_id
                )
                file_hash = self.generate_file_hash(annotation_file_path)

                self.annotation_cache.put(annotation_file_path, annotations)
                self.annotation_cache.put(f"{annotation_file_path}_hash", file_hash)

        annotations = self.annotation_cache.get(annotation_file_path)
        # Filter annotations based on the requested range
        annotationTemp = None
        filtered_annotations = []

        for annotationTemp in annotations:
            startLineAnnotation = annotationTemp["lineNumber"]
            numberOfLinesAnnotation = annotationTemp["lineCount"]
            endOfAnnotation = startLineAnnotation + numberOfLinesAnnotation - 1
            endOfRequestedInterval = startLine + numberOfLines - 1

            newstartOfAnnotation = max(startLineAnnotation, startLine)
            newEndOfAnnotation = min(endOfAnnotation, endOfRequestedInterval)

            if newstartOfAnnotation <= newEndOfAnnotation:
                annotationToAppend = annotationTemp
                annotationToAppend["lineNumber"] = newstartOfAnnotation
                annotationToAppend["lineCount"] = (
                    newEndOfAnnotation - newstartOfAnnotation + 1
                )

                filtered_annotations.append(annotationToAppend)
            else:
                if startLineAnnotation > startLine:
                    break

        return filtered_annotations

    def generate_file_hash(self, file_path: str) -> str:
        try:
            # Get the last modification time of the file
            mod_time = os.path.getmtime(file_path)
            # Create a unique string combining file path and modification time
            unique_string = f"{file_path}:{mod_time}"
            # Generate a hash using SHA-256
            file_hash = hashlib.sha256(unique_string.encode()).hexdigest()
            return file_hash
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path} - {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating file hash: {e}")
            raise

    def get_annotation_file_path(self, fileName: str, annotation_id: int) -> str:
        import Constants

        annotationFilePath = Constants.GetAnnotationsDirectoryPath(
            fileName, self.userId
        )

        fileNameWithoutExtension = os.path.splitext(os.path.basename(fileName))[0]

        return os.path.join(
            annotationFilePath,
            fileNameWithoutExtension + "_annotations_" + str(annotation_id),
        )


if __name__ == "__main__":
    annotation_id0 = 0
    file_path = "log1.log"
    annotationTestSet1 = [
        {
            "lineNumber": 2,
            "lineCount": 3,
            "AnnotationId": 0,
            "Comment": "This is a comment",
        },
        {
            "lineNumber": 7,
            "lineCount": 1,
            "AnnotationId": 0,
            "Comment": "This is a comment",
        },
        {
            "lineNumber": 17,
            "lineCount": 2,
            "AnnotationId": 0,
            "Comment": "This is a comment",
        },
        {
            "lineNumber": 2046,
            "lineCount": 3,
            "AnnotationId": 0,
            "Comment": "This is a comment",
        },
        {
            "lineNumber": 10000,
            "lineCount": 3,
            "AnnotationId": 0,
            "Comment": "This is a comment",
        },
    ]

    annotation_id1 = 1
    annotationTestSet2 = [
        {
            "lineNumber": 3,
            "lineCount": 3,
            "AnnotationId": 1,
            "Comment": "This is a comment",
        },
        {
            "lineNumber": 7,
            "lineCount": 2,
            "AnnotationId": 1,
            "Comment": "This is a comment",
        },
        {
            "lineNumber": 18,
            "lineCount": 2,
            "AnnotationId": 1,
            "Comment": "This is a comment",
        },
        {
            "lineNumber": 2047,
            "lineCount": 3,
            "AnnotationId": 1,
            "Comment": "This is a comment",
        },
        {
            "lineNumber": 10001,
            "lineCount": 3,
            "AnnotationId": 1,
            "Comment": "This is a comment",
        },
    ]
    annotationCacheManager = AnnotationCacheManager(max_cache_size=3, userId=0)
    annotationCacheManager.write_annotation_file(
        file_path=file_path,
        annotations=annotationTestSet1,
        annotation_id=annotation_id0,
    )

    annotationCacheManager = AnnotationCacheManager(max_cache_size=3, userId=0)
    annotationCacheManager.write_annotation_file(
        file_path=file_path,
        annotations=annotationTestSet2,
        annotation_id=annotation_id1,
    )

    annotationCacheManager = AnnotationCacheManager(max_cache_size=3, userId=0)
    verificationAnnotation = annotationCacheManager.read_annotation_file(
        file_path=file_path, annotation_id=annotation_id0
    )

    if verificationAnnotation == annotationTestSet1:
        print("Test 1 Passed")
    else:
        print("Test 1 Failed")

    annotationCacheManager = AnnotationCacheManager(max_cache_size=3, userId=0)

    verificationAnnotation = annotationCacheManager.get_annotations(
        file_path, startLine=0, numberOfLines=30, annotation_id=annotation_id0
    )
    if verificationAnnotation == annotationTestSet1[0:3]:
        print("Test 2 Passed")
    else:
        print("Test 2 Failed")

    verificationAnnotation = annotationCacheManager.get_annotations(
        file_path, startLine=2, numberOfLines=3, annotation_id=annotation_id0
    )
    if verificationAnnotation == annotationTestSet1[0:1]:
        print("Test 3 Passed")
    else:
        print("Test 3 Failed")

    verificationAnnotation = annotationCacheManager.get_annotations(
        file_path, startLine=7, numberOfLines=1, annotation_id=annotation_id0
    )
    if verificationAnnotation == [annotationTestSet1[1]]:
        print("Test 4 Passed")
    else:
        print("Test 4 Failed")

    verificationAnnotation = annotationCacheManager.get_annotations(
        file_path, startLine=7, numberOfLines=6, annotation_id=annotation_id0
    )
    if verificationAnnotation == annotationTestSet1[1:2]:
        print("Test 5 Passed")
    else:
        print("Test 5 Failed")

    verificationAnnotation = annotationCacheManager.get_annotations(
        file_path, startLine=700, numberOfLines=1000, annotation_id=annotation_id0
    )
    if verificationAnnotation == []:
        print("Test 6 Passed")
    else:
        print("Test 6 Failed")

    verificationAnnotation = annotationCacheManager.get_annotations(
        file_path, startLine=0, numberOfLines=1, annotation_id=annotation_id0
    )
    if verificationAnnotation == []:
        print("Test 7 Passed")
    else:
        print("Test 7 Failed")

    verificationAnnotation = annotationCacheManager.get_annotations(
        file_path, startLine=3, numberOfLines=1, annotation_id=annotation_id0
    )
    if verificationAnnotation == [
        {
            "lineNumber": 3,
            "lineCount": 1,
            "AnnotationId": annotation_id0,
            "Comment": "This is a comment",
        }
    ]:
        print("Test 8 Passed")
    else:
        print("Test 8 Failed")

    verificationAnnotation = annotationCacheManager.get_annotations(
        file_path, startLine=16, numberOfLines=5, annotation_id=annotation_id0
    )
    if verificationAnnotation == [
        {
            "lineNumber": 17,
            "lineCount": 2,
            "AnnotationId": annotation_id0,
            "Comment": "This is a comment",
        }
    ]:
        print("Test 9 Passed")
    else:
        print("Test 9 Failed")

    verificationAnnotation = annotationCacheManager.get_annotations(
        file_path, startLine=2000, numberOfLines=3000, annotation_id=annotation_id0
    )
    if verificationAnnotation == [
        {
            "lineNumber": 2046,
            "lineCount": 3,
            "AnnotationId": annotation_id0,
            "Comment": "This is a comment",
        }
    ]:
        print("Test 10 Passed")
    else:
        print("Test 10 Failed")
