import numpy as np
from threading import Lock
import os
import hashlib
import Constants
import LRUCache


class OffsetArrayManager:
    def __init__(self, max_cache_size=100, userId=0):
        self.offset_cache = LRUCache.LRUCache(max_cache_size)
        self.cache_lock = Lock()
        self.userId = userId

    def get_offsets_array(self, file_path):
        with self.cache_lock:  # Ensures thread safety
            offset_base_file_name = os.path.basename(file_path)

            offset_file_path = Constants.GetOffsetsDirectoryPath(self.userId)
            offset_file_path = (
                os.path.join(offset_file_path, offset_base_file_name) + ".offsets"
            )

            cached_file_path = self.offset_cache.get(f"{offset_file_path}_offset_file_path")   
            numberOfLines = self.offset_cache.get(f"{offset_file_path}_numberOfLines")
            offset_file_exists = os.path.exists(offset_file_path)

            if cached_file_path is None:
                if offset_file_exists:
                    log_file_mtime = os.path.getmtime(file_path)
                    offset_file_path_mtime = os.path.getmtime(offset_file_path) if os.path.exists(offset_file_path) else 0
                    
                    if log_file_mtime < offset_file_path_mtime:
                        print("[CreateSimularityFilesOfLogFile] Skipping similarity calculation as the output files exist and are newer than the log file.")
                    else:
                        self.generate_offset_file(file_path, offset_file_path)
                else:
                    self.generate_offset_file(file_path, offset_file_path)

                # Update cache with new offsets and file hash
                offset_file_content = self.read_offset_file(offset_file_path)
                numberOfLines = self.get_number_of_lines_in_file(offset_file_path)

                self.offset_cache.put(f"{offset_file_path}_offset_file_path", file_path)
                self.offset_cache.put(f"{offset_file_path}_numberOfLines", numberOfLines)
                self.offset_cache.put(offset_file_path, offset_file_content)

            return self.offset_cache.get(offset_file_path), numberOfLines

    def get_number_of_lines_in_file(self, file_path):
        file_size = os.path.getsize(file_path)  # Get file size in bytes
        # Assuming average line size of 8 bytes
        return file_size // 8

    def generate_offset_file(self, file_path, offset_file_path):
        """Generate a file containing byte offsets for each line in a binary-read file."""
        numberOfLines = 0
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            os.makedirs(os.path.dirname(offset_file_path), exist_ok=True)

            with (
                open(file_path, "rb") as f,
                open(offset_file_path, "wb") as offset_file,
            ):
                current_position = 0
                while True:
                    line = f.readline()
                    if not line:
                        break
                    numberOfLines += 1
                    # Write the current byte offset as an 8-byte little-endian integer
                    position = current_position.to_bytes(8, byteorder="little")
                    offset_file.write(position)
                    # Update the current position by the length of the line read
                    current_position += len(line)
        except (IOError, OSError) as e:
            print(f"An error occurred while accessing files: {e}")

        check_numberOfLines = self.get_number_of_lines_in_file(offset_file_path)

        if numberOfLines != check_numberOfLines:
            print(f"Number of lines mismatch: {numberOfLines} vs {check_numberOfLines}")
            exit(1)

        return numberOfLines

    def read_offset_file(self, offset_file_path):
        """Read the offset file into a Python array."""
        offsets = []
        with open(offset_file_path, "rb") as f:
            while True:
                byte = f.read(8)
                if not byte:
                    break
                offset = int.from_bytes(byte, "little")
                offsets.append(offset)
        return np.array(offsets, dtype=int)

    def generate_file_hash(self, file_path):
        """
        Generates a hash based on the file path and its last modification date.

        Parameters:
            file_path (str): The path to the file.

        Returns:
            str: A hexadecimal hash string.
        """
        try:
            # Get the last modification time of the file
            mod_time = os.path.getmtime(file_path)
            # Create a unique string combining file path and modification time
            unique_string = f"{file_path}:{mod_time}"
            # Generate a hash using SHA-256
            file_hash = hashlib.sha256(unique_string.encode()).hexdigest()
            return file_hash
        except FileNotFoundError:
            return "Error: File not found."
        except Exception as e:
            return f"Error: {e}"
