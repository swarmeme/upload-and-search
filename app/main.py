from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db import init_db
from app.processing import resume_processing_jobs
from app.search import router as search_router
from app.uploads import router as uploads_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    resume_processing_jobs()
    yield


app = FastAPI(
    title="Large File Semantic Search",
    version="1.0.0",
    description="Resumable disk-backed uploads with bounded semantic indexing.",
    lifespan=lifespan,
)
app.include_router(uploads_router)
app.include_router(search_router)


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
