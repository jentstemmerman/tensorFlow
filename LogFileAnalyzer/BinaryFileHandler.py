import json
import struct
import os

class BinaryDataFile:
    HEADER_SIZE = 512  # Fixed header space

    # Mapping for user-friendly types
    TYPE_MAP = {
        "int32": "i",
        "float32": "f",
        "float64": "d"
    }

    def __init__(self, filename, data_type="int32", numbers_per_line=4):
        """
        Initialize the binary file handler with a JSON header.

        :param filename: Path to the binary file
        :param data_type: User-friendly data type ('int32', 'float32', 'float64')
        :param numbers_per_line: How many numbers per line
        """
        if data_type not in self.TYPE_MAP:
            raise ValueError(f"Unsupported data type: {data_type}. Use one of {list(self.TYPE_MAP.keys())}")

        self.filename = filename
        self.data_type = self.TYPE_MAP[data_type]  # Convert to struct format
        self.numbers_per_line = numbers_per_line
        self.fmt = f"{numbers_per_line}{self.data_type}"  # Example: "4f" for four float32 values
        self.record_size = struct.calcsize(self.fmt)

        # Ensure file has a JSON header
        if not os.path.exists(self.filename) or os.path.getsize(self.filename) < self.HEADER_SIZE:
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
        self._write_header({"record_count": 0, "version": 1.0, "description": "Binary data file"})

    def append(self, data):
        """
        Append a new line of numbers to the binary file.

        :param data: A tuple or list of numbers matching the format
        """
        if len(data) != self.numbers_per_line:
            raise ValueError(f"Data must contain exactly {self.numbers_per_line} elements.")
        
        # Read the current header
        header = self._read_header()
        record_count = header.get("record_count", 0)

        with open(self.filename, "ab") as f:
            f.write(struct.pack(self.fmt, *data))

        # Update the record count in the header
        header["record_count"] = record_count + 1
        self._write_header(header)

    def read_next(self, file_obj):
        """
        Read the next record from an open file object.

        :param file_obj: A binary file object (opened in "rb" mode)
        :return: A tuple of numbers, or None if end of file is reached
        """
        data = file_obj.read(self.record_size)
        return struct.unpack(self.fmt, data) if data else None

    def read_at(self, index):
        """
        Read a record at a specific index (0-based).

        :param index: Record index to read
        :return: A tuple of numbers
        """
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
        return self._read_header()

# Example Usage
if __name__ == "__main__":
    binary_file = BinaryDataFile("data_with_json_header.bin", data_type="f", numbers_per_line=3)

    # Clear file (resets header)
    binary_file.clear_file()

    # Append records
    binary_file.append((1.1, 2.2, 3.3))
    binary_file.append((4.4, 5.5, 6.6))

    # Read back at index 1
    print(binary_file.read_at(1))  # Output: (4.4, 5.5, 6.6)

    # Read header
    print(binary_file.get_header_info())  # Output: {'record_count': 2, 'version': 1.0, 'description': 'Binary data file'}

    # Read sequentially
    with open("data_with_json_header.bin", "rb") as f:
        f.seek(BinaryDataFile.HEADER_SIZE)  # Skip header
        print(binary_file.read_next(f))  # Output: (1.1, 2.2, 3.3)
        print(binary_file.read_next(f))  # Output: (4.4, 5.5, 6.6)
