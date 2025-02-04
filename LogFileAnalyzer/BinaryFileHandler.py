import struct
import json
import os

class BinaryDataFile:
    HEADER_SIZE = 128  # Assuming header size is 128 bytes

    # Mapping for user-friendly types
    TYPE_MAP = {
        "int32": "i",
        "float32": "f",
        "float64": "d"
    }

    def __init__(self, filename, data_type="float32", numbers_per_line=3):
        self.filename = filename
        self.data_type = data_type
        self.numbers_per_line = numbers_per_line
        self.record_size = struct.calcsize(self.TYPE_MAP[data_type]) * numbers_per_line
        self.fmt = self.TYPE_MAP[data_type] * numbers_per_line
        self.file_handler = None
        self._write_header({"record_count": 0, "version": 1.0, "description": "Binary data file"})

    def _write_header(self, header_data):
        """Write the JSON header to the file."""
        json_header = json.dumps(header_data).encode("utf-8")
        if len(json_header) > self.HEADER_SIZE:
            raise ValueError("Header data too large! Increase HEADER_SIZE.")
        
        padded_header = json_header.ljust(self.HEADER_SIZE, b'\x00')  # Pad with null bytes
        with open(self.filename, "r+b" if os.path.exists(self.filename) else "wb") as f:
            f.seek(0)
            f.write(padded_header)

    def _read_header(self):
        """Read and return the JSON header from the file."""
        with open(self.filename, "rb") as f:
            header_bytes = f.read(self.HEADER_SIZE)
            header_str = header_bytes.rstrip(b'\x00').decode("utf-8")  # Remove padding
            return json.loads(header_str)

    def clear_file(self):
        """Clear the binary file and reset the JSON header."""
        with open(self.filename, "wb") as f:
            header = json.dumps({"record_count": 0, "version": 1.0, "description": "Binary data file"}).ljust(self.HEADER_SIZE)
            f.write(header.encode('utf-8'))

    def append(self, record):
        """Append a record to the binary file."""
        with open(self.filename, "ab") as f:
            f.write(struct.pack(self.fmt, *record))
        header = self._read_header()
        header["record_count"] += 1
        self._write_header(header)

    def read_next(self):
        """Read the next record from the file."""
        if self.file_handler is None:
            self.file_handler = open(self.filename, "rb")
            self.file_handler.seek(self.HEADER_SIZE)  # Skip header

        data = self.file_handler.read(self.record_size)
        return struct.unpack(self.fmt, data) if data else None

    def read_at(self, index):
        """Read a record at a specific index (0-based)."""
        header = self._read_header()
        record_count = header.get("record_count", 0)

        if index >= record_count:
            raise IndexError("Index out of range.")

        with open(self.filename, "rb") as f:
            f.seek(self.HEADER_SIZE + index * self.record_size)  # Adjust for header
            data = f.read(self.record_size)
            if len(data) != self.record_size:
                raise ValueError("Invalid index or file corruption.")
            return struct.unpack(self.fmt, data)

    def get_header_info(self):
        """Return the JSON header as a dictionary."""
        with open(self.filename, "rb") as f:
            header = f.read(self.HEADER_SIZE).decode('utf-8').rstrip('\x00')
            return json.loads(header)

    def close(self):
        """Close the file handler if it's open."""
        if self.file_handler:
            self.file_handler.close()
            self.file_handler = None

# Example Usage
if __name__ == "__main__":
    binary_file = BinaryDataFile("data_with_json_header.bin", data_type="float32", numbers_per_line=3)

    # Clear file (resets header)
    binary_file.clear_file()

    # Append records
    binary_file.append((1.1, 2.2, 3.3))
    binary_file.append((4.4, 5.5, 6.6))
    binary_file.append((7.7, 8.8, 9.9))

    # Read back at index 1
    print(binary_file.read_at(1))  # Output: (4.4, 5.5, 6.6)

    # Read header
    print(binary_file.get_header_info())  # Output: {'record_count': 3, 'version': 1.0, 'description': 'Binary data file'}

    # Read sequentially
    print(binary_file.read_next())  # Output: (1.1, 2.2, 3.3)
    print(binary_file.read_next())  # Output: (4.4, 5.5, 6.6)
    print(binary_file.read_next())  # Output: (7.7, 8.8, 9.9)

    # Close the file handler
    binary_file.close()
