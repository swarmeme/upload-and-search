from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
MODEL_CACHE_DIR = BASE_DIR / "model_cache"
DATABASE_URL = f"sqlite:///{DATA_DIR / 'app.db'}"

# These limits keep the service's working set bounded even for multi-GB files.
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024
MAX_UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024
READ_BUFFER_BYTES = 1024 * 1024
EMBEDDING_DIMENSIONS = 384
TEXT_CHUNK_CHARS = 1_500
CHUNK_OVERLAP_CHARS = 150
BOUNDARY_LOOKAHEAD_CHARS = 250
EMBEDDING_BATCH_SIZE = 64
MAX_CONCURRENT_PROCESSORS = 2
MODEL_NAME = "all-MiniLM-L6-v2"
