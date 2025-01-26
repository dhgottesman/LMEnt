import os
import gzip
import glob
import jsonlines

DEFAULT_MAX_CHUNK_SIZE_BYTES = 1024 * 1024 * 1024 # 1 GB

class JsonWriter:
    def __init__(self, base_path, chunk_test_interval=10000, max_chunk_size_bytes=DEFAULT_MAX_CHUNK_SIZE_BYTES, verbose=False):
        self.base_path = base_path

        self.chunk_num = 0 
        self.num_writes = 0

        self.verbose = verbose
        self.chunk_test_interval = chunk_test_interval
        self.max_chunk_size_bytes = max_chunk_size_bytes
        self.last_processed_doc = -1
        self._last_chunk()
        self._open_chunk()

    def _last_chunk(self):
        """Finds the number of the last existing chunk."""
        files = glob.glob(f'{self.base_path}*')
        self.chunk_num = len(files)
        if self.chunk_num > 0:
            self.chunk_num -= 1  # Decrement to get the last existing chunk

    def _open_chunk(self):
        # Check if the temporary file already exists and open it in append mode if it does
        if os.path.exists(self.temporary_path):
            with gzip.open(self.temporary_path, 'rb') as f:
                self.num_writes = 0
                for line in jsonlines.Reader(f):
                    self.num_writes += 1 # Count existing lines
                self.last_processed_doc = line.get("id")
            self.file = gzip.open(self.temporary_path, 'at')  # Open in append mode
        else:
            self.file = gzip.open(self.temporary_path, 'wt')
        self.writer = jsonlines.Writer(self.file, flush=True)

    def _log(self, msg):
        if self.verbose:
            print(msg)

    @property
    def path(self):
        return f"{self.base_path}_{self.chunk_num}.jsonl.gz"

    @property
    def temporary_path(self):
        return self.path + ".tmp"

    def write(self, data):
        self.writer.write(data)

        self.num_writes += 1

        if self.num_writes % self.chunk_test_interval == 0:
            self._log("Flushing and checking chunk size")
            self.file.flush()

            if os.path.getsize(self.temporary_path) >= self.max_chunk_size_bytes:
                self._log("Performing chunk rollover")
                self.close()
                self.chunk_num += 1
                self._open_chunk()

    def cleanup(self):
        self.file.flush()
        self.writer.close()
        self.file.close()

    def close(self):
        self.cleanup()
        os.rename(self.temporary_path, self.path)