import os
import io
from cStringIO import StringIO

class FileOutstream(object):
    # buffer_size: kb
    def __init__(self, file_path, buffer_size = io.DEFAULT_BUFFER_SIZE):
        self.file_path = file_path
        self.buffer_size = buffer_size * 1000
        self.buffer_stream = StringIO()
        self.num_bytes = 0
        if os.path.isfile(self.file_path):
            os.remove(self.file_path)

    def flush(self):
        if self.buffer_stream.closed:
            return
        with io.open(self.file_path, "ab") as f:
            f.write(self.buffer_stream.getvalue())
        self.buffer_stream.close()

    def write(self, s):
        if self.buffer_stream.closed:
            del self.buffer_stream
            self.buffer_stream = StringIO()
        self.buffer_stream.write(s)
        self.num_bytes += len(s)
        if self.num_bytes >= self.buffer_size:
            self.flush()

    def writeline(self, s):
        self.write(s + "\n")

    def __del__(self):
        self.flush()
