import os
import gzip
import jsonlines

DEFAULT_MIN_CHUNK_SIZE_BYTES = 1024 * 1024 * 1024 # 1 GB

class JsonWriter:
    def __init__(self, base_path, chunk_test_interval=10000, min_chunk_size_bytes=DEFAULT_MIN_CHUNK_SIZE_BYTES, verbose=False):
        self.base_path = base_path

        self.chunk_num = 0
        self.num_writes = 0

        self.verbose = verbose
        self.chunk_test_interval = chunk_test_interval
        self.min_chunk_size_bytes = min_chunk_size_bytes

        self._open_chunk()

    def _open_chunk(self):
        self.file = gzip.open(self.temporary_path, 'wt')
        self.writer = jsonlines.Writer(self.file)

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

            if os.path.getsize(self.temporary_path) >= self.min_chunk_size_bytes:
                self._log("Performing chunk rollover")
                self.close()
                self.chunk_num += 1
                self._open_chunk()


    def close(self):
        self.writer.close()
        self.file.close()
        os.rename(self.temporary_path, self.path)