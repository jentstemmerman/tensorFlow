import struct
import json
import os
import numpy as np
import tensorflow as tf

class BinaryFileHandler:
    HEADER_SIZE = 256  # Assuming header size is 128 bytes

    # Mapping for user-friendly types
    TYPE_MAP = {
        "int32": "i",
        "float32": "f",
        "float64": "d"
    }

    def __init__(self, filename, data_type="float32", numbers_per_line=3, modelHash=""):
        self.filename = filename
        self.modelHash = modelHash
        self.data_type = data_type
        self.numbers_per_line = numbers_per_line
        self.read_file_handler = None
        self.write_file_handler = None
        self._header_cache = {}
        self._read_header()
        self.numbers_per_line = self._header_cache.get("numbers_per_line", self.numbers_per_line)
        self.record_size = struct.calcsize(self.TYPE_MAP[data_type]) * self.numbers_per_line
        self.fmt = self.TYPE_MAP[data_type] * self.numbers_per_line
        
    def _write_header(self, header_data, flush=False):
        """Write the JSON header to the file."""
                
        json_header = json.dumps(header_data).encode("utf-8")
        if len(json_header) > self.HEADER_SIZE:
            raise ValueError("Header data too large! Increase HEADER_SIZE.")
        
        padded_header = json_header.ljust(self.HEADER_SIZE, b'\x00')  # Pad with null bytes

        if self.write_file_handler is None:
            self.write_file_handler = open(self.filename, "wb")
        
        self.write_file_handler.seek(0)
        self.write_file_handler.write(padded_header)
        if flush:
            self.write_file_handler.flush()

    def _read_header(self):
        self._header_cache = {}
        if not os.path.exists(self.filename):
            self._header_cache = {"record_count": 0, "version": 1.0, "description": "Binary data file", "numbers_per_line": self.numbers_per_line, "modelHash": self.modelHash}
            self._write_header(self._header_cache, flush=True)
            
        if self.read_file_handler is None:
            self.read_file_handler = open(self.filename, "rb")
            
            header_bytes = self.read_file_handler.read(self.HEADER_SIZE)
            header_str = header_bytes.rstrip(b'\x00').decode("utf-8")  # Remove padding
            self._header_cache = json.loads(header_str)
            
            return self._header_cache

    def get_modelHash(self):
        header = self._get_header()
        self.modelHash = header.get("modelHash", 0)
        
        return self.modelHash


    def get_numbersPerLine(self):
        return self.numbers_per_line
    
    def _get_header(self):
        """Read and return the JSON header from the file or from the member if already read."""
        if hasattr(self, '_header_cache'):
            return self._header_cache
        else:
            return self._read_header()        


    def clear_file(self):
        """Clear the binary file and reset the JSON header."""
        self.modelHash = ""
        with open(self.filename, "wb") as f:
            header = json.dumps({"record_count": 0, "version": 1.0, "description": "Binary data file", "numbers_per_line": self.numbers_per_line, "modelHash": self.modelHash}).ljust(self.HEADER_SIZE)
            f.write(header.encode('utf-8'))
            

    def append(self, record):
        """Append a record to the binary file."""
        if self.write_file_handler is None:
            self.write_file_handler = open(self.filename, "ab")
        
        # Ensure record is iterable
        if ((not isinstance(record, np.ndarray)) and (not isinstance(record, (list, tuple)))):
            record = [record]
        
        # Check for numpy array and convert to list
        if hasattr(record, 'tolist'):
            record = record.tolist()
        
        # Convert numpy.float32 to native Python float
        record = [float(x) for x in record]    # this is not needed for float32
        self.write_file_handler.write(struct.pack(self.fmt, *record))
        self._header_cache["record_count"] += 1
        #self.write_file_handler.flush()
        # Update the header in the file
        

    def read_next(self):
        """Read the next record from the file."""
        if self.read_file_handler is None:
            self.read_file_handler = open(self.filename, "rb")
            self.read_file_handler.seek(self.HEADER_SIZE)  # Skip header

        data = self.read_file_handler.read(self.record_size)
        return struct.unpack(self.fmt, data) if data else None

    def read_at(self, index):
        """Read a record at a specific index (0-based)."""
        header = self._get_header()
        record_count = header.get("record_count", 0)

        if index >= record_count:
            raise IndexError("Index out of range.")

        if self.read_file_handler is None:
            self.read_file_handler = open(self.filename, "rb")
            self.read_file_handler.seek(self.HEADER_SIZE)  # Skip header

        self.read_file_handler.seek(self.HEADER_SIZE + index * self.record_size)  # Adjust for header
        data = self.read_file_handler.read(self.record_size)
        
        # Debug statements
        print(f"Header size: {self.HEADER_SIZE}")
        print(f"Record size: {self.record_size}")
        print(f"Index: {index}")
        print(f"Data length: {len(data)}")
        print(f"Expected data length: {self.record_size}")

        if len(data) != self.record_size:
            raise ValueError("Invalid index or file corruption.")
        return struct.unpack(self.fmt, data)

    def read_records(self, start_index, num_records):
        """Read a specified number of records starting from a specific index and return as a TensorFlow tensor."""

        header = self._get_header()
        record_count = header.get("record_count", 0)

        if start_index >= record_count:
            raise IndexError("Start index out of range.")
        if start_index + num_records > record_count:
            raise IndexError("Number of records to read exceeds available records.")

        if self.read_file_handler is None:
            self.read_file_handler = open(self.filename, "rb")
            self.read_file_handler.seek(self.HEADER_SIZE)  # Skip header

        self.read_file_handler.seek(self.HEADER_SIZE + start_index * self.record_size)  # Adjust for header
        records = []
        for _ in range(num_records):
            data = self.read_file_handler.read(self.record_size)
            if len(data) != self.record_size:
                raise ValueError("Invalid index or file corruption.")
            records.append(struct.unpack(self.fmt, data))
        
        return tf.convert_to_tensor(records, dtype=tf.float32)

    def close(self):
        """Close the file handler if it's open."""
        if self.read_file_handler:
            self.read_file_handler.close()
            self.read_file_handler = None
        if self.write_file_handler:
            self._write_header(self._header_cache)
            self.write_file_handler.close()
            self.write_file_handler = None




# Example Usage
if __name__ == "__main__":
        # Delete the file if it exists
    if os.path.exists("data_with_json_header.bin"):
        os.remove("data_with_json_header.bin")
        
    # Clear file (resets header)
    
    binary_file = BinaryFileHandler("data_with_json_header.bin", data_type="float32", numbers_per_line=3)
    binary_file.clear_file()


    # Append records
    binary_file.append((1.1, 2.2, 3.3))
    binary_file.append((4.4, 5.5, 6.6))
    binary_file.append((7.7, 8.8, 9.9))
    binary_file.close()
    
    binary_file = None
    binary_file = BinaryFileHandler("data_with_json_header.bin", data_type="float32", numbers_per_line=3)
    
    # Read back at index 1
    print(binary_file.read_at(1))  # Output: (4.4, 5.5, 6.6)
    print(binary_file.read_at(0))  # Output: (4.4, 5.5, 6.6)
    # Read sequentially
    print(binary_file.read_next())  # Output: (1.1, 2.2, 3.3)
    print(binary_file.read_next())  # Output: (4.4, 5.5, 6.6)
    print(binary_file.read_next())  # Output: (7.7, 8.8, 9.9)

    # Close the file handler
    binary_file.close()
