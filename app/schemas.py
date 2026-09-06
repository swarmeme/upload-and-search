from datetime import datetime

from pydantic import BaseModel, Field

from app.models import FileStatus


class CreateFileRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=512)
    total_size: int = Field(gt=0)


class CreateFileResponse(BaseModel):
    file_id: str
    status: FileStatus
    total_size: int


class FileStatusResponse(BaseModel):
    file_id: str
    filename: str
    status: FileStatus
    total_size: int
    bytes_received: int
    next_offset: int
    bytes_indexed: int
    error_message: str | None = None
    created_at: datetime


class SearchRequest(BaseModel):
    file_id: str
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    text: str
    score: float
    byte_offset: int
    line_start: int
    line_end: int


class SearchResponse(BaseModel):
    file_id: str
    results: list[SearchResult]
