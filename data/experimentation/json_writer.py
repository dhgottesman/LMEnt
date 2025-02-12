import os
import gzip
import glob
import jsonlines

DEFAULT_MAX_CHUNK_SIZE_BYTES = 1024 * 1024 * 1024 # 1 GB

class JsonWriter:
    def __init__(self, base_path, chunk_test_interval=600000, max_chunk_size_bytes=DEFAULT_MAX_CHUNK_SIZE_BYTES, verbose=True):
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
            if os.path.exists(self.path):  # The final path was written and we are ready for a rollover
                self.chunk_num += 1

    def _open_chunk(self):
        # Check if the temporary file already exists and open it in append mode if it does
        if os.path.exists(self.temporary_path):
            with gzip.open(self.temporary_path, 'rb') as f:
                self.num_writes = 0
                for line in jsonlines.Reader(f):
                    self.num_writes += 1 # Count existing lines
                    self.last_processed_doc = line.get("id")
                # Temporary file exists but it is empty
                if self.num_writes == 0 and self.chunk_num > 0:
                    assert(os.path.exists(self.last_committed_path))
                    with gzip.open(self.last_committed_path, 'rb') as f:
                        for line in jsonlines.Reader(f):
                            self.last_processed_doc = line.get("id")
            self.file = gzip.open(self.temporary_path, 'at', errors='replace')  # Open in append mode
        elif os.path.exists(self.last_committed_path):
            with gzip.open(self.last_committed_path, 'rb') as f:
                for line in jsonlines.Reader(f):
                    self.last_processed_doc = line.get("id")
            self.file = gzip.open(self.temporary_path, 'wt', errors='replace')
        else:
            self.file = gzip.open(self.temporary_path, 'wt', errors='replace')
        self.writer = jsonlines.Writer(self.file, flush=True)

    def _log(self, msg):
        if self.verbose:
            print(msg)

    @property
    def last_committed_path(self):
        return f"{self.base_path}_{self.chunk_num - 1}.jsonl.gz"

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
            os.fsync(self.file.fileno())

            if os.path.getsize(self.temporary_path) >= self.max_chunk_size_bytes:
                self._log("Performing chunk rollover")
                self.close()
                self.chunk_num += 1
                self._open_chunk()

    def cleanup(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        self.writer.close()
        self.file.close()

    def close(self):
        self.cleanup()
        os.rename(self.temporary_path, self.path)
