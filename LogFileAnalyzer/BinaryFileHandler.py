import struct
import json
import os
import numpy as np
import tensorflow as tf

class BinaryFileHandler:
    HEADER_SIZE = 512  # Assuming header size is 128 bytes

    # Mapping for user-friendly types
    TYPE_MAP = {
        "int32": "i",
        "float32": "f",
        "float64": "d"
    }

    def __init__(self, filename, sequence_length, data_type="float32", numbers_per_line=3, modelHash=""):
        self.filename = filename
        self.sequence_length = sequence_length
        self.modelHash = modelHash
        self.data_type = data_type
        self.numbers_per_line = numbers_per_line
        self.read_file_handler = None
        self.write_file_handler = None
        self._header_cache = {}
        self._read_header()
        self.sequence_length = self._header_cache.get("sequence_length", self.sequence_length)
        self.numbers_per_line = self._header_cache.get("numbers_per_line", self.numbers_per_line)
        self.record_size = struct.calcsize(self.TYPE_MAP[data_type]) * self.numbers_per_line
        self.fmt = self.TYPE_MAP[data_type] * self.numbers_per_line
        self.set_chunk_size_by_memory(1024)
        
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
        try:
            if not os.path.exists(self.filename):
                self._header_cache = {"record_count": 0, "version": 1.0, "description": "Binary data file", "numbers_per_line": self.numbers_per_line, "modelHash": self.modelHash, "sequence_length": self.sequence_length}
                self._write_header(self._header_cache, flush=True)
                
            if self.read_file_handler is None:
                self.read_file_handler = open(self.filename, "rb")
            
            self.read_file_handler.seek(0)    
            header_bytes = self.read_file_handler.read(self.HEADER_SIZE)
            
            header_str = header_bytes.rstrip(b'\x00').decode("utf-8")  # Remove padding and ignore errors
            # header_str = header_bytes.rstrip(b'\x00').decode("utf-8", errors="ignore")  # Remove padding and ignore errors
            self._header_cache = json.loads(header_str)
        except Exception as e:
            print(f"Error reading header: {e}")
            self._header_cache = {}
        return self._header_cache

    def get_modelHash(self):
        header = self._get_header()
        self.modelHash = header.get("modelHash", 0)
        
        return self.modelHash
    
    def get_recordCount(self):
        header = self._get_header()
       
        return header.get("record_count", 0)
    
    def get_sequenceLength(self):
        header = self._get_header()
       
        return header.get("sequence_length", 0)

    def get_numbersPerLine(self):
        return self.numbers_per_line
    
    def _get_header(self):
        """Read and return the JSON header from the file or from the member if already read."""
        if not hasattr(self, '_header_cache'):

            self._header_cache = self._read_header()
        
        return self._header_cache        


    def clear_file(self):
        """Clear the binary file and reset the JSON header."""
        self.modelHash = ""
        with open(self.filename, "wb") as f:
            header = json.dumps({"record_count": 0, "sequence_length": 0, "version": 1.0, "description": "Binary data file", "numbers_per_line": self.numbers_per_line, "modelHash": self.modelHash}).ljust(self.HEADER_SIZE)
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
        
        if len(data) != self.record_size:
            return None
        
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
        print(f"sequence_length: {self.sequence_length}") 
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
        
        data = self.read_file_handler.read(self.record_size * num_records)
        if len(data) != self.record_size * num_records:
            raise ValueError("Invalid index or file corruption.")
        
        records = np.frombuffer(data, dtype=self.data_type).reshape(num_records, self.numbers_per_line)
        return tf.convert_to_tensor(records, dtype=tf.float32)

    def close(self):
        """Close the file handler if it's open."""
        if self.read_file_handler:
            self.read_file_handler.close()
            self.read_file_handler = None
        if self.write_file_handler:

            self._write_header(self._header_cache, flush=True)
            self.write_file_handler.close()
            self.write_file_handler = None

    def set_chunk_size_by_memory(self, memory_bytes):
        """
        Sets the chunk size based on a memory budget in bytes.
        It estimates the memory usage per record and then calculates the maximum
        number of records that fit into the given memory budget.
        
        :param memory_bytes: Memory budget in bytes.
        """
        # Assume each record is stored along the first dimension.
        record_shape = (self.numbers_per_line,)
        
        # If any dimension is unknown (None), assume 1 for estimation.
        dims = [dim if dim is not None else 1 for dim in record_shape]
        record_num_elements = np.prod(dims)
        
        # Get the size in bytes for one element of this dtype.
        element_size = np.dtype(self.data_type).itemsize
        record_size = record_num_elements * element_size

        # Compute how many records fit into the memory budget.
        chunk_size = memory_bytes // record_size
        if chunk_size < 1:
            chunk_size = 1  # Ensure at least one record per chunk.
        self.chunk_size = int(chunk_size)
        #print(f"Chunk size set to {self.chunk_size} records per chunk based on a memory budget of {memory_bytes} bytes.")

    def get_chunk_tensor(self, chunk_index):
        """
        Retrieves the tensor for the specified chunk.
        If the chunk is the last one, it may contain fewer records than chunk_size.
        """
        start = chunk_index * self.chunk_size
        end = start + self.chunk_size

        if self.read_file_handler is None:
            self.read_file_handler = open(self.filename, "rb")
            self.read_file_handler.seek(self.HEADER_SIZE)  # Skip header

        self.read_file_handler.seek(self.HEADER_SIZE + start * self.record_size)
        records = []
        for _ in range(self.chunk_size):
            data = self.read_file_handler.read(self.record_size)
            if len(data) != self.record_size:
                break
            records.append(struct.unpack(self.fmt, data))

        return tf.convert_to_tensor(records, dtype=tf.float32)

    def read_records_optimised(self, start_index, num_records):
        """
        Reads records in a sliding-window style, handling cases where the window spans multiple chunks.
        """
        chunk_size = self.chunk_size
        start_chunk = start_index // chunk_size
        end_index = start_index + num_records - 1
        end_chunk = end_index // chunk_size

        # Case 1: Entire window is within one chunk.
        if start_chunk == end_chunk:
            chunk_tensor = self.get_chunk_tensor(start_chunk)
            start_offset = start_index % chunk_size
            # Make sure we don't slice beyond the available records (especially for a small last chunk).
            chunk_length = tf.shape(chunk_tensor)[0]
            end_offset = tf.minimum(start_offset + num_records, chunk_length)
            return chunk_tensor[start_offset:end_offset]

        # Case 2: The window spans multiple chunks.
        parts = []

        # First chunk: from start_offset to the end of the chunk.
        first_chunk_tensor = self.get_chunk_tensor(start_chunk)
        start_offset = start_index % chunk_size
        parts.append(first_chunk_tensor[start_offset:])

        # Middle chunks: add entire chunks if they exist.
        for chunk in range(start_chunk + 1, end_chunk):
            parts.append(self.get_chunk_tensor(chunk))

        # Last chunk: from the beginning to the required number of records.
        last_chunk_tensor = self.get_chunk_tensor(end_chunk)
        end_offset = (end_index % chunk_size) + 1  # +1 because end_index is inclusive.
        last_chunk_length = tf.shape(last_chunk_tensor)[0]
        end_offset = tf.minimum(end_offset, last_chunk_length)
        parts.append(last_chunk_tensor[:end_offset])

        # Concatenate all parts along the first axis.
        return tf.concat(parts, axis=0)

# Example Usage
if __name__ == "__main__":
    # Delete the file if it exists
    if os.path.exists("data_with_json_header.bin"):
        os.remove("data_with_json_header.bin")
        
    # Clear file (resets header)
    import ModelHash
    modelHash = ModelHash.ModelHash()
    binary_file = BinaryFileHandler("data_with_json_header.bin", data_type="float32", numbers_per_line=3, modelHash=modelHash.getHash())
    header= binary_file._read_header()
    binary_file.append((1.1, 2.2, 3.3))
    header= binary_file._read_header()
    binary_file._write_header( header, flush=False)
    
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
