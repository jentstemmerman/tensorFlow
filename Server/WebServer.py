from flask import Flask, jsonify, request, send_file, g
from flask_cors import CORS
import os
import FileHandler
import Constants
import json
from flask_socketio import SocketIO, emit, disconnect
import time
import threading
import sys
import math
import numpy as np

sys.path.append('../LogFileAnalyzer')
import CommandAnalyzeFile # type: ignore
import DefaultTrainingSettings # type: ignore
import Signature # type: ignore
import uuid

from queue import Queue
from UserDB import UserDB
from AuthMiddleware import (
    require_auth, create_access_token, create_refresh_token,
    validate_socketio_token,
)

WebServer = Flask(__name__)

# CORS: configurable via CORS_ORIGINS env var (comma-separated).  Default: allow all (dev).
_cors_origins = os.environ.get("CORS_ORIGINS", "*")
if _cors_origins != "*":
    _cors_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()]
cors = CORS(WebServer, resources={r"/*": {"origins": _cors_origins}})

socketio = SocketIO(WebServer, cors_allowed_origins=_cors_origins)

# Shared user database instance
user_db = UserDB()


# --- Analyze File Queue ---------------------------------------------------------
ANALYZE_WORKER_COUNT = 1  # Increase if you want parallel workers (watch CPU/RAM)
analyze_job_queue: Queue = Queue()
analyze_job_status = {}      # jobId -> dict(status, progress, fileName, userId, error)
_analyze_status_lock = threading.Lock()


def _to_serializable(obj):
    """Recursively convert objects to JSON-serializable types.

    - NumPy arrays -> lists
    - NumPy scalars -> Python scalars
    - bytes -> utf-8 strings (replace errors)
    - tuples -> lists
    - sets -> lists
    - non-finite floats (NaN/Inf) -> None
    - dicts/lists -> recurse
    """
    # Attempt optional numpy handling without hard dependency at import time
    np_types = None
    try:
        import numpy as np  # type: ignore
        np_types = (np.ndarray, np.generic)
    except Exception:
        np_types = None

    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, set):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, float):
        # Replace non-finite floats to keep JSON valid
        return obj if math.isfinite(obj) else None
    if np_types is not None:
        if isinstance(obj, np_types[0]):  # ndarray
            return obj.tolist()
        if isinstance(obj, np_types[1]):  # numpy scalar
            try:
                return obj.item()
            except Exception:
                return str(obj)
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return str(obj)
    return obj


def _set_job_status(job_id, **updates):
    with _analyze_status_lock:
        if job_id in analyze_job_status:
            analyze_job_status[job_id].update(updates)


def update_job_and_emit(job_id: str, file_name: str, userId: str, *, progress: int, status: str, error: str | None = None):
    """Update internal job state and emit websocket progress.

    Keyword-only args for progress/status allow easy functools.partial to pre-bind everything
    except the changing progress value, e.g.:

        progress_cb = partial(update_job_and_emit, job_id, file_name, userId, status="running")
        progress_cb(progress=25)
    """
    _set_job_status(job_id, status=status, progress=progress, error=error)
    emit_analyze_file_progress(file_name, userId, progress, status=status, error=error, jobId=job_id)


def emit_analyze_file_progress(file_name: str, userId: str, progress: int, status: str = "running", error: str | None = None, jobId: str | None = None):
    try:
        payload = {
            "fileName": file_name,
            "userId": userId,  # FIX: was UserId
            "progress": progress,
            "status": status,
        }
        if jobId:
            payload["jobId"] = jobId
        if error is not None:
            payload["error"] = str(error)
        socketio.emit("analyze_file_progress", payload, to=f"user_{userId}")
    except Exception as e:
        print(f"emit_analyze_file_progress failed: {e}")


def _analyze_worker():
    while True:
        job = analyze_job_queue.get()
        if job is None:
            break
        job_id = job["jobId"]
        file_name = job["fileName"]
        userId = job["userId"]
        modelDirectory = job["modelDirectory"]
        tokenDirectory = job["tokenDirectory"]
        try:
            update_job_and_emit(job_id, file_name, userId, progress=0, status="started")

            modelDirFull = os.path.join(Constants.GetModelsDirectoryPath(), modelDirectory)
            file_path = Constants.GetLogFilePath(logFilename=file_name, userId=userId)
            OriginalTokenFileDirectory = os.path.join(Constants.GetTokenFilesPath(), tokenDirectory)
            itemFile = os.path.join(Constants.GetItemFilesPath(), "Itemfile.txt")
            LogFileAnalyzedPath = Constants.GetLogFileAnalyzedPath(logFilename=file_name, userId=userId)
            temperature = 1
            signatureInputDirectory = Constants.GetUserSpecificBaseDirectory("SignatureType", userId)
            similarityOutputDirectory = os.path.dirname(
                Constants.GetMachineReadableSimularitiesFile(file_name, userId, "")
            )
            inferenceDataSetFile = Constants.GetInferenceDataSetPath(file_path, userId, temperature)

            # Early stub progress
            # Prepare a partial so deeper layers (future) can call progress_cb(progress=..)
            from functools import partial
            progress_cb = partial(update_job_and_emit, job_id, file_name, userId, status="starting")
            progress_cb(progress=0, status="starting")

            # Real analysis
            CommandAnalyzeFile.AnalyzeFile(
                modelDirectory=modelDirFull,
                userId=userId,
                inputFile=file_path,
                inferenceDataSetFile=inferenceDataSetFile,
                LogFileAnalyzedPath=LogFileAnalyzedPath,
                signatureInputDirectory=signatureInputDirectory,
                similarityOutputDirectory=similarityOutputDirectory,
                OriginalTokenFileDirectory=OriginalTokenFileDirectory,
                itemFile=itemFile,
                temperature=temperature,
                progress_cb=progress_cb,
            )

            # Stub fixed number
            progress_cb(progress=42)

            update_job_and_emit(job_id, file_name, userId, progress=100, status="done")
        except Exception as e:
            print(f"Analyze job {job_id} error: {e}")
            update_job_and_emit(job_id, file_name, userId, progress=0, status="error", error=str(e))
        finally:
            analyze_job_queue.task_done()


def _start_analyze_workers():
    # Idempotent
    if getattr(_start_analyze_workers, "_started", False):
        return
    for _ in range(ANALYZE_WORKER_COUNT):
        t = threading.Thread(target=_analyze_worker, daemon=True)
        t.start()
    _start_analyze_workers._started = True


# NOTE: duplicate emit_analyze_file_progress removed here. The active definition with jobId support
# is located above near the queue setup.


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


# ===========================================================================
#  Authentication endpoints (public — no @require_auth)
# ===========================================================================

@WebServer.route("/api/auth/login", methods=["POST"])
def auth_login():
    """Authenticate with username+password, return JWT tokens."""
    data = request.get_json(force=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    user = user_db.authenticate(username, password)
    if user is None:
        return jsonify({"error": "Invalid credentials"}), 401
    access_token = create_access_token(user["id"], user["username"], user["role"])
    refresh_token, refresh_expires = create_refresh_token(user["id"], user_db)
    return jsonify({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "userId": user["id"],
        "username": user["username"],
        "displayName": user["display_name"],
        "role": user["role"],
    })


@WebServer.route("/api/auth/refresh", methods=["POST"])
def auth_refresh():
    """Exchange a valid refresh token for a new access token."""
    data = request.get_json(force=True) or {}
    refresh_token = data.get("refresh_token", "")
    if not refresh_token:
        return jsonify({"error": "refresh_token is required"}), 400
    user = user_db.validate_refresh_token(refresh_token)
    if user is None:
        return jsonify({"error": "Invalid or expired refresh token"}), 401
    access_token = create_access_token(user["id"], user["username"], user["role"])
    return jsonify({"access_token": access_token})


@WebServer.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    """Revoke a refresh token."""
    data = request.get_json(force=True) or {}
    refresh_token = data.get("refresh_token", "")
    if refresh_token:
        user_db.revoke_refresh_token(refresh_token)
    return jsonify({"message": "Logged out"})


@WebServer.route("/api/auth/me", methods=["GET"])
@require_auth
def auth_me():
    """Return current user info from the JWT."""
    return jsonify({
        "userId": g.user_id,
        "username": g.username,
        "role": g.user_role,
    })


# ===========================================================================
#  Protected API endpoints (all use @require_auth, userId from g.user_id)
# ===========================================================================

@WebServer.route("/api/get_directory_listing", methods=["GET"])
@require_auth
def get_directory_listing():
    """
    Return the list of files and directories for a given path.
    Supports folder browsing, including navigation to parent directories.
    """
    # Get the 'path' parameter from the request
    requested_path = request.args.get("path", "")
    requested_directoryType = request.args.get("directoryType", "")
    userId = str(g.user_id)

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

        #print("get_log_files_list: Returning the list of files and directories")
        #print(files_with_paths)

        return jsonify(files_with_paths)
    except Exception as e:
        print(f"Error listing directory contents: {e}")
        return jsonify({"error": "Failed to list directory contents"}), 500


@WebServer.route("/api/signature_files", methods=["GET"])
@require_auth
def get_signature_files_list():
    """Return the list of available signature files with their file paths."""
    print("get_files: Return the list of available text files")

    userId = str(g.user_id)

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
@require_auth
def get_file_lines(file_name):
    """Return a specific part of the file content based on line numbers."""
    try:
        start_time = time.time()
        lineOffset = int(request.args.get("start", 0))  # 0 means the default value
        numberOfLines = int(request.args.get("size", 1024))
        userId = str(g.user_id)
        similaritiesOnly = bool(int(request.args.get("similaritiesOnly", 0)))
        file_path = Constants.GetLogFilePath(logFilename=file_name, userId=userId)

        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        simularity_base_directory = os.path.dirname(Constants.GetMachineReadableSimularitiesFile(file_name, userId, ""))

        simularity_files = []

        if os.path.exists(simularity_base_directory):
            # Only include .dat files (case-insensitive), exclude packed similarity files
            # which use a raw uint64 binary format incompatible with BinaryFileHandler
            simularity_files = [
                f for f in os.listdir(simularity_base_directory)
                if f.lower().endswith('.dat')
                and '_packed_' not in f.lower()
                and os.path.isfile(os.path.join(simularity_base_directory, f))
            ]

        simularity_files_with_paths = [
            {"fileName": file, "filePath": os.path.join(simularity_base_directory, file)}
            for file in simularity_files
        ]
        
        simularitiesList = []
        
        for simularity_file in simularity_files_with_paths:
            simularities =  FileHandler.GetSimularityData(simularity_file["filePath"], lineOffset, numberOfLines, DefaultTrainingSettings.TrainingsParameter["DataTypeForTensors"])
            simularityData = {}
            simularityData["simularity_file"] = simularity_file["fileName"]
            
            if simularities is None:
                simularityData["simularities"] = []
            else:
                # Downcast to float16 for transmission to reduce payload size and precision
                # We do NOT change the on-disk storage dtype; only the JSON response is quantized.
                try:
                    simularities = np.asarray(simularities).astype(np.float16, copy=False)
                except Exception:
                    # If casting fails for any reason, fall back to original
                    pass
                # Ensure pure Python types for JSON
                simularityData["simularities"] = _to_serializable(simularities)
            simularitiesList.append(simularityData)

        if (similaritiesOnly is False):
            content = FileHandler.read_file_lines(file_path, userId, lineOffset, numberOfLines)
            content = content.decode("utf-8", errors="replace")
        else:
            content = ""
            
        elapsed = time.time() - start_time
        
        print(f"get_file_lines file_name: lineOffset: {lineOffset}, numberOfLines: {numberOfLines} - took {elapsed:.4f}s")
        
        # Verify that the returned content has the expected number of lines
        actual_line_count = len(content.splitlines())
        if actual_line_count != numberOfLines:
            nl = content.count('\n')
            print(f"Warning: requested {numberOfLines} lines, but returned {actual_line_count} lines, counted {nl} newlines")
        
        payload = {
            "text": content,
            "start": lineOffset,
            "size": numberOfLines,
            "simularities": simularitiesList
        }

        # Sanitize to JSON-safe types and encode deterministically
        safe_payload = _to_serializable(payload)
        body = json.dumps(safe_payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return WebServer.response_class(body, mimetype="application/json; charset=utf-8")
    
    except Exception as e:
        print(e)
        try:
            return jsonify({"error": str(e)}), 500
        except Exception as enc_err:
            # Fallback if jsonify fails due to bad data
            print(f"JSON encode failed: {enc_err}")
            return WebServer.response_class(
                json.dumps({"error": "Serialization failed"}, ensure_ascii=False),
                status=500,
                mimetype="application/json; charset=utf-8",
            )


@WebServer.route("/api/file/analyze_file/<file_name>", methods=["GET"])
@require_auth
def analyze_file(file_name):
    userId = str(g.user_id)
    modelDirectory = request.args.get("modelDirectory")
    tokenDirectory = request.args.get("tokenDirectory")

    if not all([userId, modelDirectory, tokenDirectory]):
        return jsonify({"error": "Missing required parameters"}), 400

    job_id = str(uuid.uuid4())
    job = {
        "jobId": job_id,
        "fileName": file_name,
        "userId": userId,  # FIX: was UserId
        "modelDirectory": modelDirectory,
        "tokenDirectory": tokenDirectory,
        "created": time.time(),
    }

    with _analyze_status_lock:
        analyze_job_status[job_id] = {
            "status": "queued",
            "progress": 0,
            "fileName": file_name,
            "userId": userId,  # FIX: was UserId
            "error": None,
        }

    queue_position = analyze_job_queue.qsize()
    analyze_job_queue.put(job)

    return jsonify({
        "accepted": True,
        "jobId": job_id,
        "queuePosition": queue_position,
    })


@WebServer.route("/api/file/analyze_file/status/<job_id>", methods=["GET"])
@require_auth
def analyze_file_status(job_id):
    with _analyze_status_lock:
        status = analyze_job_status.get(job_id)
    if not status:
        return jsonify({"error": "Unknown jobId"}), 404
    return jsonify({"jobId": job_id, **status})


@WebServer.route("/api/file/checkannotations/<file_name>", methods=["GET"])
@require_auth
def check_file_annotations(file_name):
    try:
        userId = str(g.user_id)
        annotation_id = str(request.args.get("annotation_id", "0"))

        annotation_id = os.path.splitext(annotation_id)[0]
        
        annotations_exists = FileHandler.CheckAnnotations(
            file_name, userId, annotation_id
        )

        return jsonify({"annotations_exists": annotations_exists})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@WebServer.route("/api/file/getannotations/<file_name>/<signature_name>", methods=["GET"])
@require_auth
def get_file_annotations(file_name, signature_name):
    """
    Returns the packed similarity file as raw binary (application/octet-stream).
    The binary format is a flat array of uint64 pairs: (lineNumber, numLines).
    Each record is 16 bytes (2 x 8-byte little-endian unsigned integers).
    On the client, parse with: new BigUint64Array(await response.arrayBuffer())
    """
    try:
        userId = str(g.user_id)

        packed_file_path = Constants.GetPackedMachineReadableSimularitiesFile(
            file_name, userId, signature_name
        )

        if not os.path.exists(packed_file_path):
            return jsonify({"error": "Packed similarity file not found"}), 404

        return send_file(
            packed_file_path,
            mimetype="application/octet-stream",
            as_attachment=False,
        )

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@WebServer.route("/api/file/meta/<file_name>", methods=["GET"])
@require_auth
def get_file_metadata(file_name):
    """Return metadata for a given file, such as total file size."""

    userId = str(g.user_id)

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
@require_auth
def upload_file():
    print("/api/uploadLogFile ")
    file = request.files["file"]
    userId = str(g.user_id)
    baseDirectory = Constants.GetUserSpecificBaseDirectory("LogType", userId)
    file_path = os.path.join(baseDirectory, file.filename)

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    file.save(file_path)
    return jsonify({"message": "File uploaded successfully", "filePath": file_path})

@WebServer.route('/api/createFile', methods=['POST'])
@require_auth
def create_file():
    data = request.json
    
    user_id = str(g.user_id)
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
    
@WebServer.route("/api/createSignatureFile", methods=["POST"])
@require_auth
def upload_signature():
    print("/api/createSignatureFile")

    signature_name = request.form["signature_name"]
    logFileName = request.form["logFileName"]
    selectedTextForSignature = json.loads(request.form["selectedTextForSignature"])
    
    userId = str(g.user_id)
    baseDirectory = Constants.GetUserSpecificBaseDirectory("LogType", userId)
    
    logFileName = os.path.join(baseDirectory, logFileName)
    Signature.Signature.SaveSignatureFile(signature_name, logFileName, selectedTextForSignature, userId)

    if signature_name is not None:
        return jsonify(
        {
            "message": "Signature uploaded successfully",
            "signature_name": signature_name,
        }
        )
    else:
        return jsonify({"error": "Failed to upload signature"}), 500



@WebServer.route("/api/file/repack_similarity", methods=["POST"])
@require_auth
def repack_similarity():
    """Re-pack a single signature's similarity data with a new threshold.

    Expects JSON body: { fileName, signatureName, threshold }
    Runs the SimilarityPacker in a background thread and emits a
    ``repack_complete`` Socket.IO event when finished.
    """
    try:
        data = request.get_json(force=True)
        file_name = data.get("fileName")
        user_id = str(g.user_id)
        signature_name = data.get("signatureName")
        threshold = float(data.get("threshold", -0.8))

        if not file_name or not signature_name:
            return jsonify({"error": "fileName and signatureName are required"}), 400

        DataTypeForTensors = DefaultTrainingSettings.TrainingsParameter["DataTypeForTensors"]

        def _background_repack():
            try:
                count = FileHandler.RepackSimilarityFile(
                    file_name, user_id, signature_name, threshold, DataTypeForTensors
                )
                socketio.emit("repack_complete", {
                    "fileName": file_name,
                    "userId": user_id,
                    "signatureName": signature_name,
                    "threshold": threshold,
                    "intervalCount": count,
                    "success": True,
                    "error": None,
                }, to=f"user_{user_id}")
            except Exception as e:
                print(f"Error repacking similarity: {e}")
                socketio.emit("repack_complete", {
                    "fileName": file_name,
                    "userId": user_id,
                    "signatureName": signature_name,
                    "threshold": threshold,
                    "intervalCount": 0,
                    "success": False,
                    "error": str(e),
                }, to=f"user_{user_id}")

        threading.Thread(target=_background_repack, daemon=True).start()
        return jsonify({"accepted": True})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@WebServer.route("/api/PostProcessLogFiles", methods=["GET"])
@require_auth
def PostProcessLogFiles():
    try:
        fileName = request.args.get("fileName", None)
        userId = str(g.user_id)
        similaritythreshold = float(request.args.get("similaritythreshold", 0.5))
        
        def background_task(fileName, userId, DataTypeForTensors, similaritythreshold):
            try:
                if fileName is None:
                    FileHandler.PostProcessLogFiles(userId, DataTypeForTensors, similaritythreshold)
                else:
                    FileHandler.PostProcessLogFile(fileName, userId, DataTypeForTensors, similaritythreshold)
            except Exception as e:
                print(f"Error post processing log files for user {userId}: {e}")
                #socketio.emit('preprocess_error', {'error': str(e), 'userId': userId})

        # Start a background thread to process the log files
        threading.Thread(target=background_task, args=(fileName, userId, DefaultTrainingSettings.TrainingsParameter["DataTypeForTensors"],similaritythreshold,)).start()

        return jsonify({"started": True})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@WebServer.route("/api/settings", methods=["GET"])
@require_auth
def get_user_settings():
    user_id = str(g.user_id)
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


@WebServer.route("/api/settings", methods=["POST"])
@require_auth
def save_user_settings():
    user_id = str(g.user_id)
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

    _start_analyze_workers()
    socketio.run(WebServer, host="0.0.0.0", port=5000, debug=True)


@socketio.on("connect")
def handle_connect(auth=None):
    """Authenticate Socket.IO connections via JWT in auth parameter."""
    payload = validate_socketio_token(auth)
    if payload is None:
        print("Socket.IO connection rejected: invalid token")
        disconnect()
        return
    # Join a user-specific room so events can be targeted
    from flask_socketio import join_room
    user_room = f"user_{payload['sub']}"
    join_room(user_room)
    print(f"Client connected: user={payload.get('username')} room={user_room}")
    emit("message", {"data": "Connected to server"})


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")
