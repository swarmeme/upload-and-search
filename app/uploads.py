import asyncio
import os
from collections import defaultdict

from fastapi import APIRouter, Header, HTTPException, Request, status
from sqlalchemy import select

from app.config import MAX_FILE_SIZE, MAX_UPLOAD_CHUNK_BYTES, UPLOAD_DIR
from app.db import SessionLocal
from app.models import FileStatus, StoredFile, UploadRange
from app.schemas import CreateFileRequest, CreateFileResponse, FileStatusResponse

router = APIRouter(prefix="/files", tags=["files"])
_upload_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def upload_path(file_id: str):
    """Only the server-generated UUID is ever used as a filesystem name."""
    return UPLOAD_DIR / f"{file_id}.upload"


def next_missing_offset(session, file_id: str) -> int:
    """Return the end of the contiguous prefix beginning at byte zero."""
    cursor = 0
    ranges = session.scalars(
        select(UploadRange).where(UploadRange.file_id == file_id).order_by(UploadRange.start_offset)
    )
    for interval in ranges:
        if interval.start_offset > cursor:
            break
        cursor = max(cursor, interval.end_offset)
    return cursor


def status_payload(session, stored: StoredFile) -> FileStatusResponse:
    return FileStatusResponse(
        file_id=stored.id,
        filename=stored.original_filename,
        status=stored.status,
        total_size=stored.total_size,
        bytes_received=stored.bytes_received,
        next_offset=next_missing_offset(session, stored.id),
        bytes_indexed=stored.bytes_indexed,
        error_message=stored.error_message,
        created_at=stored.created_at,
    )


@router.post("", response_model=CreateFileResponse, status_code=status.HTTP_201_CREATED)
async def create_upload(request: CreateFileRequest) -> CreateFileResponse:
    if request.total_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"Files may be at most {MAX_FILE_SIZE} bytes")

    stored = StoredFile(original_filename=request.filename, total_size=request.total_size)
    with SessionLocal.begin() as session:
        session.add(stored)
        session.flush()  # assigns the UUID before touching the filesystem
        file_id = stored.id

    # truncate reserves the final logical length without allocating 10 GB in RAM.
    try:
        with open(upload_path(file_id), "wb") as destination:
            destination.truncate(request.total_size)
            destination.flush()
            os.fsync(destination.fileno())
    except OSError as exc:
        with SessionLocal.begin() as session:
            failed = session.get(StoredFile, file_id)
            if failed:
                failed.status = FileStatus.FAILED
                failed.error_message = f"Could not allocate upload file: {exc}"
        raise HTTPException(status_code=507, detail="Could not allocate storage for this upload") from exc

    return CreateFileResponse(file_id=file_id, status=FileStatus.UPLOADING, total_size=request.total_size)


@router.patch("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def write_chunk(
    file_id: str,
    request: Request,
    upload_offset: int = Header(..., alias="Upload-Offset", ge=0),
) -> None:
    """Write a raw request body at its absolute offset, tus-style."""
    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="Upload chunk must not be empty")
    if len(payload) > MAX_UPLOAD_CHUNK_BYTES:
        raise HTTPException(status_code=413, detail=f"Chunks may be at most {MAX_UPLOAD_CHUNK_BYTES} bytes")

    async with _upload_locks[file_id]:
        with SessionLocal() as session:
            stored = session.get(StoredFile, file_id)
            if not stored:
                raise HTTPException(status_code=404, detail="Upload session not found")
            if stored.status != FileStatus.UPLOADING:
                raise HTTPException(status_code=409, detail=f"File is {stored.status.value}; it no longer accepts chunks")
            if upload_offset + len(payload) > stored.total_size:
                raise HTTPException(status_code=416, detail="Chunk exceeds declared file size")

        try:
            # r+b plus seek/write supports retries and out-of-order chunks. fsync makes
            # the data durable before the database acknowledges it.
            with open(upload_path(file_id), "r+b") as destination:
                destination.seek(upload_offset)
                destination.write(payload)
                destination.flush()
                os.fsync(destination.fileno())
        except OSError as exc:
            raise HTTPException(status_code=507, detail="Could not persist upload chunk") from exc

        should_enqueue = False
        with SessionLocal.begin() as session:
            stored = session.get(StoredFile, file_id)
            if not stored or stored.status != FileStatus.UPLOADING:
                raise HTTPException(status_code=409, detail="Upload session changed while writing")

            start, end = upload_offset, upload_offset + len(payload)
            touching = list(
                session.scalars(
                    select(UploadRange).where(
                        UploadRange.file_id == file_id,
                        UploadRange.start_offset <= end,
                        UploadRange.end_offset >= start,
                    )
                )
            )
            for interval in touching:
                start = min(start, interval.start_offset)
                end = max(end, interval.end_offset)
                session.delete(interval)
            session.add(UploadRange(file_id=file_id, start_offset=start, end_offset=end))
            session.flush()

            # Intervals are merged, so their lengths are exactly the unique bytes received.
            stored.bytes_received = sum(
                interval.end_offset - interval.start_offset
                for interval in session.scalars(select(UploadRange).where(UploadRange.file_id == file_id))
            )
            if stored.bytes_received == stored.total_size:
                stored.status = FileStatus.PROCESSING
                stored.error_message = None
                should_enqueue = True

    if should_enqueue:
        from app.processing import enqueue_processing

        enqueue_processing(file_id)


@router.get("/{file_id}", response_model=FileStatusResponse)
async def get_file_status(file_id: str) -> FileStatusResponse:
    with SessionLocal() as session:
        stored = session.get(StoredFile, file_id)
        if not stored:
            raise HTTPException(status_code=404, detail="File not found")
        return status_payload(session, stored)
