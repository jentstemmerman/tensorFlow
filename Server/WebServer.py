from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import FileHandler
import Constants
import json
from flask_socketio import SocketIO, emit
import time
import threading
import sys
sys.path.append('../LogFileAnalyzer')
import CommandAnalyzeFile # type: ignore
import DefaultTrainingSettings # type: ignore
import SignatureMatching # type: ignore
import AnnotationManager # type: ignore
import shutil

WebServer = Flask(__name__)
cors = CORS(WebServer, resources={r"/*": {"origins": "*"}})

socketio = SocketIO(WebServer, cors_allowed_origins=["*"])


# Mock annotations
TextAnnotations = {
    "file1.txt": [{"line": 2, "label": "Important"}, {"line": 5, "label": "To-Do"}],
    "file2.txt": [{"line": 3, "label": "Note"}, {"line": 4, "label": "Warning"}],
}

# Mock annotations
ColumnAnnotations = [
    {"lineNumber": 2, "lineCount": "3", "color": "red"},
    {"lineNumber": 7, "lineCount": "1", "color": "red"},
    {"lineNumber": 17, "lineCount": "2", "color": "green"},
]

ColumnAnnotations = [
    {
        "lineNumber": 2,
        "lineCount": "3",
        "AnnotationId": 1,
        "Comment": "This is a comment",
    },
    {
        "lineNumber": 7,
        "lineCount": "1",
        "AnnotationId": 2,
        "Comment": "This is a comment",
    },
    {
        "lineNumber": 17,
        "lineCount": "2",
        "AnnotationId": 1,
        "Comment": "This is a comment",
    },
]


def GetOffsetFilePath(LogFilename, userId):
    baseDirectory = Constants.GetUserSpecificBaseDirectory("OffsetsType", userId)

    offset_file_path = os.path.join(baseDirectory, LogFilename)
    return offset_file_path


@WebServer.route("/api/get_directory_listing", methods=["GET"])
def get_directory_listing():
    """
    Return the list of files and directories for a given path.
    Supports folder browsing, including navigation to parent directories.
    """
    print("get_directory_listing: Returning the list of files and directories")

    # Get the 'path' parameter from the request
    requested_path = request.args.get("path", "")
    requested_directoryType = request.args.get("directoryType", "")
    userId = request.args.get("userId", "")

    base_directory = ""

    class ServerDirectoryType:
        LogFileType = "LogType"
        SignatureDirType = "SignatureType"
        ModelDirType = "ModelDirType"

    base_directory = Constants.GetUserSpecificBaseDirectory(requested_directoryType, userId)

    # Resolve the full directory path
    full_directory_path = os.path.normpath(os.path.join(base_directory, requested_path))

    # Ensure the directory is within the root directory
    if not full_directory_path.startswith(base_directory):
        return jsonify({"error": "Access denied"}), 403

    if not os.path.exists(full_directory_path):
        os.makedirs(full_directory_path)

    try:
        entries = os.listdir(full_directory_path)
        files_with_paths = []

        # Add a ".." entry for navigating to the parent directory if not at the root
        if full_directory_path != os.path.normpath(base_directory):
            parent_path = os.path.relpath(
                os.path.dirname(full_directory_path), base_directory
            )
            files_with_paths.append(
                {
                    "fileName": "..",
                    "filePath": parent_path if parent_path != "." else "",
                    "isDirectory": True,
                }
            )

        # Process the directory entries
        for entry in entries:
            full_path = os.path.join(full_directory_path, entry)
            files_with_paths.append(
                {
                    "fileName": entry,
                    "filePath": full_path,  # Relative path for client
                    "isDirectory": os.path.isdir(full_path),
                }
            )

        print("get_log_files_list: Returning the list of files and directories")
        print(files_with_paths)

        return jsonify(files_with_paths)
    except Exception as e:
        print(f"Error listing directory contents: {e}")
        return jsonify({"error": "Failed to list directory contents"}), 500


@WebServer.route("/api/signature_files", methods=["GET"])
def get_signature_files_list():
    """Return the list of available signature files with their file paths."""
    print("get_files: Return the list of available text files")

    userId = request.args.get("userId", "")

    base_directory = Constants.GetSignatureDirectoryPath(userId)

    files = []

    if os.path.exists(base_directory):
        files = os.listdir(base_directory)

    files_with_paths = [
        {"fileName": file, "filePath": os.path.join(base_directory, file)}
        for file in files
    ]

    return jsonify(files_with_paths)


@WebServer.route("/api/file/getlines/<file_name>", methods=["GET"])
def get_file_lines(file_name):
    """Return a specific part of the file content based on line numbers."""
    try:
        lineOffset = int(request.args.get("start", 0))  # 0 means the default value
        numberOfLines = int(request.args.get("size", 1024))
        userId = str(request.args.get("userId", 0))
        file_path = Constants.GetLogFilePath(logFilename=file_name, userId=userId)
        # print("get_file_lines file_name: ", file_name, ", lineOffset: ", lineOffset, "numberOfLines: ", numberOfLines, "offset_file_path: ", offset_file_path, "file_path: ", file_path)

        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        content = FileHandler.read_file_lines(
            file_path, userId, lineOffset, numberOfLines
        )
        
        content = content.decode("utf-8", errors="ignore")

        return jsonify(
            {
                "text": content,
                "start": lineOffset,
                "size": numberOfLines,
               # "annotations": ColumnAnnotations,
            }
        )
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

analyze_file_busy = False

@WebServer.route("/api/file/analyze_file/<file_name>", methods=["GET"])
def analyze_file(file_name):
    """Return a specific part of the file content based on line numbers."""
    
    global analyze_file_busy
    
    if analyze_file_busy:
        print("Busy. END")
        return jsonify({"error": True}), 500
    
    analyze_file_busy = True
    
    try:
        userId = request.args.get("userId", None)
        modelDirectory = request.args.get("modelDirectory", None)
        tokenDirectory = request.args.get("tokenDirectory", None)
        modelDirectory = os.path.join(Constants.GetModelsDirectoryPath(), modelDirectory)
        
        file_path = Constants.GetLogFilePath(logFilename=file_name, userId=userId)
        
        OriginalTokenFileDirectory =  os.path.join( Constants.GetTokenFilesPath(), tokenDirectory)
            
        itemFile = os.path.join( Constants.GetItemFilesPath(), "Itemfile.txt")
                
        LogFileAnalyzedPath = Constants.GetLogFileAnalyzedPath(logFilename=file_name, userId=userId)
        
        temperature = 1
         
        FileNameMachineReadablePooledVectors = Constants.GetLogFileMachineReadableVectorsPath(file_name, userId, temperature)
        FileNameHumanReadablePooledVectors = Constants.GetLogFileHumanReadableVectorsPath(file_name, userId, temperature)
        
        # Clear directories
        directories_to_clear = [
            os.path.dirname(FileNameMachineReadablePooledVectors),
            os.path.dirname(FileNameHumanReadablePooledVectors)
        ]

        for directory in directories_to_clear:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                os.makedirs(directory)
        
        
        CommandAnalyzeFile.AnalyzeFile(modelDirectory, file_path, LogFileAnalyzedPath, FileNameMachineReadablePooledVectors, FileNameHumanReadablePooledVectors, OriginalTokenFileDirectory, itemFile, temperature=1)

        # print("get_file_lines file_name: ", file_name, ", lineOffset: ", lineOffset, "numberOfLines: ", numberOfLines, "offset_file_path: ", offset_file_path, "file_path: ", file_path)

        return jsonify(
            {
                "ok": 1,
            }
        )
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@WebServer.route("/api/file/checkannotations/<file_name>", methods=["GET"])
def check_file_annotations(file_name):
    try:
        userId = request.args.get("userId", None)
        annotation_id = int(request.args.get("annotation_id", 0))

        annotations_exists = FileHandler.CheckAnnotations(
            file_name, userId, annotation_id
        )

        return jsonify({"annotations_exists": annotations_exists})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@WebServer.route("/api/file/getannotations/<file_name>", methods=["GET"])
def get_file_annotations(file_name):
    """Return a specific part of the file content based on line numbers."""
    try:
        lineOffset = int(request.args.get("start", 0))  # 0 means the default value
        userId = request.args.get("userId", None)
        numberOfLines = int(request.args.get("size", 1024))
        annotation_id = int(request.args.get("annotation_id", 0))

        print(
            "[App.py] get_file_annotations file_name: ",
            file_name,
            "userId: ",
            userId,
            ", lineOffset: ",
            lineOffset,
            "numberOfLines: ",
            numberOfLines,
            "annotation_id: ",
            annotation_id,
        )

        # if not os.path.exists(file_path):
        #    return jsonify({"error": "File not found"}), 404

        annotations = FileHandler.GetAnnotations(
            file_name, userId, lineOffset, numberOfLines, annotation_id
        )

        return jsonify({"annotations": annotations})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@WebServer.route("/api/file/meta/<file_name>", methods=["GET"])
def get_file_metadata(file_name):
    """Return metadata for a given file, such as total file size."""

    userId = request.args.get("userId", None)

    file_path = Constants.GetLogFilePath(file_name, userId)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    numberOfLines = FileHandler.GetNumberOfLinesInFile(file_path, userId)

    try:
        file_size = os.path.getsize(file_path)

        return jsonify({"size": file_size, "numberOfLines": numberOfLines})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@WebServer.route("/api/uploadLogFile", methods=["POST"])
def upload_file():
    print("/api/uploadLogFile ")
    file = request.files["file"]
    userId = request.form["userId"]
    baseDirectory = Constants.GetUserSpecificBaseDirectory("LogType", userId)
    file_path = os.path.join(baseDirectory, file.filename)

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    file.save(file_path)
    return jsonify({"message": "File uploaded successfully", "filePath": file_path})

@WebServer.route('/api/createFile', methods=['POST'])
def create_file():
    data = request.json
    
    user_id = data.get('userId')
    path = data.get('path')
    file_name = data.get('fileName')
    file_type = data.get('fileType')
    contents = data.get('contents')
    
    baseDirectory = Constants.GetUserSpecificBaseDirectory(file_type, user_id)

    # Construct the full file path
    full_path = os.path.join(baseDirectory, path, file_name)

    try:
        # Create the file and write contents
        with open(full_path, 'w') as f:
            f.write(contents)  # Write the contents to the file

        return jsonify({"message": "File created successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@WebServer.route("/api/uploadSignatureFile", methods=["POST"])
def upload_signature():
    print("/api/uploadSignatureFile")

    try:
        signature_name = request.form["signature_name"]
        logFileName = request.form["logFileName"]
        signature_text = request.form["signature_text"]
        startLine = int(request.form["startLine"])
        endLine = int(request.form["endLine"])
        userId = request.form["userId"]
        modelDirectory = request.form["modelDirectory"]

        
        print("uploadSignature: ",
            "signature_name: ",
            signature_name,

            "userId: ",
            userId,
            "modelDirectory: ",
            modelDirectory,
            "logFileName: ",
            logFileName,
            "signature_text: ",
            signature_text,
            "startLine: ",
            startLine,
            "endLine: ",
            endLine,
        )
        
        temperature = 1
        Constants.LOG_FILE_ANALYZER_DIRECTORY
        FileNameMachineReadablePooledVectors = Constants.GetLogFileMachineReadableVectorsPath(logFileName, userId, temperature)
        DataTypeForTensors = DefaultTrainingSettings.TrainingsParameter["DataTypeForTensors"]
        modelDirectory = os.path.join(Constants.GetModelsDirectoryPath(), modelDirectory)
   
        # the modelHash is taken from the header of the machineReadablePooledVectors file
        signature, modelHash = SignatureMatching.CalculateSignature(FileNameMachineReadablePooledVectors, startLine, endLine - startLine + 1, DataTypeForTensors)

        # Pack all these values into a json and save that to the file
        signature_data = {
            "modelDirectory": modelDirectory,
            "signature_name": signature_name,
            "logFileName": logFileName,
            "signature_text": signature_text,
            "startLine": startLine,
            "endLine": endLine,
            "DataTypeForTensors" : DataTypeForTensors,
            "modelHash": modelHash,
            "signature": signature.numpy().tolist(),
        }

       
        baseDirectory = Constants.GetUserSpecificBaseDirectory("SignatureType", userId)
        file_path = os.path.join(baseDirectory, signature_name)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            json.dump(signature_data, file)

        print("Signature saved. modelHash: ", modelHash) 
       
        return jsonify(
            {
                "message": "Signature uploaded successfully",
                "signature_name": signature_name,
            }
        )

    except Exception as e:
        print("Error uploading signature:", e)
        return jsonify({"message": "Error uploading signature", "error": str(e)}), 500

@WebServer.route("/api/enableSignature", methods=["POST"])
def enable_signature():
    print("api/enableSignature")

    try:
        signature_name = request.form["signature_name"]
        logFileName = request.form["logFileName"]
        userId = request.form["userId"]
        path = request.form["path"]
        similaritythreshold = float(request.form["similarityThreshold"])
        
        print("enable_signature: "
            "signature_name: ",
            signature_name,
            "similaritythreshold: ",
            similaritythreshold,
            "path: ",
            path,
            "userId: ",
            userId,
            "logFileName: ",
            logFileName,
        )
        
        baseDirectory = Constants.GetSignatureDirectoryPath(userId)
        signature_file = os.path.join(baseDirectory, signature_name)
        simularitiesMachineReadableFileName = Constants.GetLogFileMachineReadableSimularitiesFile(logFileName, userId, signature_name)
        simularitiesHumanReadableFileName = Constants.GetLogFileHumanReadableSimularitiesFile(logFileName, userId, signature_name)
        DataTypeForTensors = DefaultTrainingSettings.TrainingsParameter["DataTypeForTensors"]
        
        similarityTresholdCrossedCounter = SignatureMatching.CreateSimularitiesFile(logFileName, signature_file, simularitiesMachineReadableFileName, simularitiesHumanReadableFileName, userId, similaritythreshold, DataTypeForTensors)
        annotation_id = 0
        outputFile = AnnotationManager.AnnotationCacheManager.get_annotation_file_path(logFileName, annotation_id, userId)
        SignatureMatching.CreateAnnotationListFromSimularitiesFile(simularitiesMachineReadableFileName, outputFile, DataTypeForTensors, annotationId = annotation_id, threshold = 0.8, comment = "")
        
        return jsonify(
            {
                "message": "Signature uploaded successfully",
                "similarityTresholdCrossedCounter": similarityTresholdCrossedCounter,
                "signature_name": signature_name,
            }
        )

    except Exception as e:
        print("Error enable_signature:", e)
        return jsonify({"message": "Error enable_signature", "error": str(e)}), 500



@WebServer.route("/api/settings/<user_id>", methods=["GET"])
def get_user_settings(user_id):
    settings_file_path = Constants.GetSettingsDirectoryPath(user_id)

    settings_file_path = os.path.join(
        settings_file_path, Constants.USER_SETTINGS_FILE_NAME
    )

    if os.path.exists(settings_file_path):
        with open(settings_file_path, "r") as file:
            settings = json.load(file)
        return jsonify({"error": "", "settings": settings})
    else:
        return jsonify({"error": "Settings file not found"}), 404


@WebServer.route("/api/settings/<user_id>", methods=["POST"])
def save_user_settings(user_id):
    settings = request.json.get("settings", {})
    settings_file_path = Constants.GetSettingsDirectoryPath(user_id)
    settings_file_path = os.path.join(
        settings_file_path, Constants.USER_SETTINGS_FILE_NAME
    )

    os.makedirs(os.path.dirname(settings_file_path), exist_ok=True)
    with open(settings_file_path, "w") as file:
        json.dump(settings, file)
    return jsonify({"message": "Settings saved successfully"})




@WebServer.route("/")
def home():
    port = request.environ.get("SERVER_PORT", "5000")
    return f"Hello, lost soul! You are connected on port {port}"


if __name__ == "__main__":
    import logging

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    def send_stubbed_data():
        while True:
            time.sleep(5)  # Send data every 5 seconds
            socketio.emit(
                "update", {"data": "This is a stubbed update from the server"}
            )

    # Start a background thread to send stubbed data
    threading.Thread(target=send_stubbed_data).start()

    socketio.run(WebServer, host="0.0.0.0", port=5000, debug=True)


@socketio.on("connect")
def handle_connect():
    print("Client connected")
    emit("message", {"data": "Connected to server"})


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")
