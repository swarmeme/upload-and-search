"""Bounded, restartable streaming text indexing."""

import asyncio
import codecs
import re
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer
from sqlalchemy import delete, select, text

from app.config import (
    BOUNDARY_LOOKAHEAD_CHARS,
    CHUNK_OVERLAP_CHARS,
    EMBEDDING_BATCH_SIZE,
    MAX_CONCURRENT_PROCESSORS,
    MODEL_CACHE_DIR,
    MODEL_NAME,
    READ_BUFFER_BYTES,
    TEXT_CHUNK_CHARS,
)
from app.db import SessionLocal
from app.models import FileStatus, StoredFile, TextChunk
from app.uploads import upload_path

_processor_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROCESSORS)
_queued_or_running: set[str] = set()
_background_tasks: set[asyncio.Task] = set()
_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    """Load the CPU model once, rather than once per upload worker."""
    global _model
    with _model_lock:
        if _model is None:
            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _model = SentenceTransformer(MODEL_NAME, cache_folder=str(MODEL_CACHE_DIR), device="cpu")
    return _model


def _choose_boundary(value: str) -> int:
    """Pick a natural break near the target, falling back to a character boundary."""
    window = value[:TEXT_CHUNK_CHARS]
    minimum = TEXT_CHUNK_CHARS // 2
    candidates: list[int] = []
    for marker in ("\n\n", "\n"):
        position = window.rfind(marker)
        if position >= minimum:
            candidates.append(position + len(marker))
    candidates.extend(match.end() for match in re.finditer(r"[.!?](?:\s+|$)", window) if match.end() >= minimum)
    return max(candidates, default=TEXT_CHUNK_CHARS)


def _stream_chunks(
    path: Path, start_offset: int, start_line: int, total_size: int
) -> Iterator[tuple[str, int, int, int, bool]]:
    """Yield (text, byte offset, first line, next checkpoint, is_last) without whole-file reads.

    The incremental decoder preserves incomplete UTF-8 sequences between reads. Byte
    offsets are advanced from encoded complete characters only, so a checkpoint is
    always a valid UTF-8 boundary.
    """
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    buffer = ""
    buffer_offset = start_offset
    buffer_line = start_line

    def emit(cut: int, final: bool = False):
        nonlocal buffer, buffer_offset, buffer_line
        chunk = buffer[:cut]
        line_start = buffer_line
        if final:
            return chunk, buffer_offset, line_start, total_size, True
        # Retain a character-based overlap. This keeps both ends UTF-8 safe.
        advance_chars = max(1, cut - CHUNK_OVERLAP_CHARS)
        advance = buffer[:advance_chars]
        next_offset = buffer_offset + len(advance.encode("utf-8"))
        next_line = buffer_line + advance.count("\n")
        buffer = buffer[advance_chars:]
        buffer_offset, buffer_line = next_offset, next_line
        return chunk, buffer_offset - len(advance.encode("utf-8")), line_start, next_offset, False

    with path.open("rb") as source:
        source.seek(start_offset)
        while data := source.read(READ_BUFFER_BYTES):
            buffer += decoder.decode(data, final=False)
            while len(buffer) >= TEXT_CHUNK_CHARS + BOUNDARY_LOOKAHEAD_CHARS:
                yield emit(_choose_boundary(buffer))
        buffer += decoder.decode(b"", final=True)
        if buffer:
            # The tail is intentionally emitted even if it is smaller than the target.
            yield emit(len(buffer), final=True)


def _write_batch(file_id: str, batch: list[tuple[str, int, int, int, bool]], total_size: int) -> None:
    model = _get_model()
    texts = [item[0] for item in batch]
    embeddings = model.encode(
        texts,
        batch_size=len(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    checkpoint = total_size if batch[-1][4] else batch[-1][3]
    checkpoint_line = batch[-1][2] + batch[-1][0].count("\n") if batch[-1][4] else _line_after_batch(batch[-1])

    # Chunk rows and their vec0 rows are committed with the checkpoint in one SQLite
    # transaction. A crash therefore resumes at a point with no half-acknowledged batch.
    with SessionLocal.begin() as session:
        stored = session.get(StoredFile, file_id)
        if not stored or stored.status != FileStatus.PROCESSING:
            return
        for item, embedding in zip(batch, embeddings, strict=True):
            chunk_id = str(uuid.uuid4())
            chunk_text, byte_offset, line_start, _next_offset, _is_last = item
            session.add(
                TextChunk(
                    id=chunk_id,
                    file_id=file_id,
                    byte_offset=byte_offset,
                    line_start=line_start,
                    line_end=line_start + chunk_text.count("\n"),
                    text=chunk_text,
                )
            )
            session.execute(
                text("INSERT INTO vec_embeddings (embedding, chunk_id, file_id) VALUES (:embedding, :chunk_id, :file_id)"),
                {
                    "embedding": np.asarray(embedding, dtype=np.float32).tobytes(),
                    "chunk_id": chunk_id,
                    "file_id": file_id,
                },
            )
        stored.indexed_checkpoint = checkpoint
        stored.checkpoint_line = checkpoint_line
        stored.bytes_indexed = checkpoint
        if checkpoint == total_size:
            stored.status = FileStatus.INDEXED
            stored.error_message = None


def _line_after_batch(item: tuple[str, int, int, int, bool]) -> int:
    """The next chunk starts after the emitted text minus its retained overlap."""
    chunk_text, _offset, line_start, _next_offset, _is_last = item
    advance_chars = max(1, len(chunk_text) - CHUNK_OVERLAP_CHARS)
    return line_start + chunk_text[:advance_chars].count("\n")


def index_file(file_id: str) -> None:
    """Synchronous worker body, deliberately run outside FastAPI's event loop."""
    try:
        with SessionLocal() as session:
            stored = session.get(StoredFile, file_id)
            if not stored or stored.status != FileStatus.PROCESSING:
                return
            checkpoint, line, total_size = stored.indexed_checkpoint, stored.checkpoint_line, stored.total_size

        batch: list[tuple[str, int, int, int, bool]] = []
        for item in _stream_chunks(upload_path(file_id), checkpoint, line, total_size):
            batch.append(item)
            if len(batch) == EMBEDDING_BATCH_SIZE:
                _write_batch(file_id, batch, total_size)
                batch.clear()
        if batch:
            _write_batch(file_id, batch, total_size)
        elif checkpoint == total_size:
            with SessionLocal.begin() as session:
                stored = session.get(StoredFile, file_id)
                if stored and stored.status == FileStatus.PROCESSING:
                    stored.status = FileStatus.INDEXED
    except Exception as exc:  # Persist a useful error instead of losing the upload session.
        with SessionLocal.begin() as session:
            stored = session.get(StoredFile, file_id)
            if stored:
                stored.status = FileStatus.FAILED
                stored.error_message = f"Indexing failed: {type(exc).__name__}: {exc}"[:2_000]


async def _run_queued_job(file_id: str) -> None:
    try:
        async with _processor_semaphore:
            await asyncio.to_thread(index_file, file_id)
    finally:
        _queued_or_running.discard(file_id)


def enqueue_processing(file_id: str) -> None:
    """Queue at most one local worker per file; the semaphore limits active jobs to two."""
    if file_id in _queued_or_running:
        return
    _queued_or_running.add(file_id)
    task = asyncio.create_task(_run_queued_job(file_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def resume_processing_jobs() -> None:
    """Called at startup to continue jobs that were durable-but-incomplete on restart."""
    with SessionLocal() as session:
        pending = session.scalars(select(StoredFile.id).where(StoredFile.status == FileStatus.PROCESSING)).all()
    for file_id in pending:
        enqueue_processing(file_id)
