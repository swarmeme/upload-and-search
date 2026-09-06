import asyncio

import numpy as np
from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from app.db import SessionLocal
from app.models import FileStatus, StoredFile, TextChunk
from app.processing import _get_model
from app.schemas import SearchRequest, SearchResponse, SearchResult

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest) -> SearchResponse:
    with SessionLocal() as session:
        stored = session.get(StoredFile, request.file_id)
        if not stored:
            raise HTTPException(status_code=404, detail="File not found")
        if stored.status != FileStatus.INDEXED:
            raise HTTPException(
                status_code=409,
                detail=f"File is {stored.status.value}; semantic search is available after indexing completes",
            )

        query_embedding = await asyncio.to_thread(
            _get_model().encode,
            [request.query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        nearest = session.execute(
            text(
                "SELECT chunk_id, distance FROM vec_embeddings "
                "WHERE embedding MATCH :embedding AND file_id = :file_id AND k = :top_k "
                "ORDER BY distance"
            ),
            {
                "embedding": np.asarray(query_embedding, dtype=np.float32).tobytes(),
                "file_id": request.file_id,
                "top_k": request.top_k,
            },
        ).all()
        ids = [row.chunk_id for row in nearest]
        if not ids:
            return SearchResponse(file_id=request.file_id, results=[])

        chunks = session.query(TextChunk).filter(TextChunk.id.in_(ids)).all()
        by_id = {chunk.id: chunk for chunk in chunks}
        results = []
        for row in nearest:
            chunk = by_id.get(row.chunk_id)
            if chunk is None:
                continue
            # Vectors are L2-normalized. For unit vectors cosine similarity is 1-d²/2.
            score = 1.0 - (float(row.distance) ** 2 / 2.0)
            results.append(
                SearchResult(
                    text=chunk.text,
                    score=score,
                    byte_offset=chunk.byte_offset,
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                )
            )
        return SearchResponse(file_id=request.file_id, results=results)
