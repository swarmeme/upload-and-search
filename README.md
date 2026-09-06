# Large File Semantic Search

A FastAPI take-home service that accepts resumable, positional uploads up to 10 GB and semantically searches a completed UTF-8 text file. It is intentionally local-first: metadata and vectors live in SQLite, and source bytes stay on the local disk.

## Project layout

```
large-file-semantic-search/
├── app/
│   ├── config.py       # Resource limits, paths, model/chunking constants
│   ├── db.py           # SQLite/SQLAlchemy setup and sqlite-vec registration
│   ├── models.py       # Persistent files, upload ranges, and text chunks
│   ├── schemas.py      # Validated request/response contracts
│   ├── uploads.py      # Upload session creation, durable positional chunk writes
│   ├── processing.py   # Bounded background stream/chunk/embed/checkpoint worker
│   ├── search.py       # Query embedding and sqlite-vec nearest-neighbour lookup
│   └── main.py         # FastAPI wiring and restart recovery
├── data/               # Created at runtime: SQLite database and UUID-named uploads
├── model_cache/        # Created at runtime: downloaded sentence-transformers model
├── requirements.txt
└── .gitignore
```

`schemas.py` is a deliberate small addition to the suggested skeleton: keeping HTTP validation separate from SQLAlchemy persistence models makes both modules simpler to read. Upload files are named from a server-generated UUID, never from the user-supplied filename.

## Run locally

Use Python 3.11 or newer (3.11–3.13 is the most broadly supported range for the ML dependencies):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The interactive API documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). On first indexing/search, `sentence-transformers` downloads `all-MiniLM-L6-v2` into the ignored `model_cache/` directory.

## API walkthrough

Start a session:

```bash
curl -X POST http://127.0.0.1:8000/files \
  -H 'content-type: application/json' \
  -d '{"filename":"notes.txt","total_size":12345}'
```

The response contains `file_id`. Send raw chunks at absolute offsets (8 MB maximum per request):

```bash
curl -X PATCH "http://127.0.0.1:8000/files/FILE_ID" \
  -H 'Upload-Offset: 0' \
  --data-binary @first-part.bin
```

Check `/files/FILE_ID` at any time. `bytes_received` counts unique durable byte coverage; `next_offset` is the first missing byte in the contiguous prefix and is the offset a normal sequential client should resume from. Once every byte is covered, status becomes `processing`, then `indexed`.

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H 'content-type: application/json' \
  -d '{"file_id":"FILE_ID","query":"what are the treatment options?","top_k":5}'
```

Search is deliberately rejected with HTTP 409 while a file is `uploading`, `processing`, or `failed`; this avoids presenting incomplete retrieval as an empty answer.

## Design discussion

### 1. How does a 10 GB file fit on a 4 GB RAM machine?

It never does. A session pre-allocates a disk file to its final length and each request writes only its 8 MB-or-smaller body at a requested offset. The indexer reads a 1 MB disk buffer at a time, keeps only a roughly 1,750-character text window and one 64-item embedding batch, then commits the batch immediately. The model is loaded once and only two indexers may run at a time.

### 2. How are interrupted uploads handled?

The client starts a session once, then uses `PATCH` plus `Upload-Offset` for absolute writes. A retry overwrites the same disk interval safely; out-of-order chunks are valid too. After `flush` and `fsync` complete, the service merges that byte interval into durable SQLite range metadata. `GET /files/{id}` reports the first contiguous missing offset, so a sequential client can restart from exactly where it stopped after a process restart.

### 3. How are multiple concurrent uploads supported?

Each session has a UUID, an isolated disk file, and its own database records. The only in-process coordination is a per-file async lock around range bookkeeping, avoiding an interval update race for two chunks of the *same* file. Different files can upload independently; SQLite WAL mode improves concurrent local reads/writes. Processing, rather than uploads, is bounded to two jobs to protect memory.

### 4. How are large files processed and indexed efficiently?

The worker uses Python's incremental UTF-8 decoder, so a multibyte character split across disk reads is completed on the next read rather than corrupted. It chooses paragraph, line, or sentence endings near 1,500 characters and retains 150 characters of overlap. It embeds up to 64 chunks at a time, writes text metadata and `sqlite-vec` rows in one transaction, then persists a byte-and-line checkpoint. On startup, `processing` jobs are queued again from that checkpoint instead of starting over.

### 5. How does semantic search work?

Both text chunks and the user query use the CPU `all-MiniLM-L6-v2` model, producing normalized 384-dimensional vectors. `sqlite-vec` performs the nearest-neighbour lookup scoped to the requested file. The relational `text_chunks` table stores the returned text plus byte offset and line range; the response converts normalized L2 distance to cosine similarity (`1 - distance²/2`).

### 6. What changes at thousands of concurrent uploads and searches?

Local disk and one SQLite database are intentionally the boundary of this exercise. I would move bytes to object storage (for example S3/GCS multipart uploads), place durable processing jobs on a message queue, and horizontally scale stateless workers. Search would move to a distributed vector system such as Qdrant or Milvus, or FAISS with IVF+PQ compression where operational ownership is acceptable; metadata would use a service-grade relational database. Upload status and idempotency records would remain durable and independently scalable.

## AI tools used

Codex was used to help implement this take-home service.

Personal review/changes: _TODO: describe the implementation details you reviewed, tested, or changed._
