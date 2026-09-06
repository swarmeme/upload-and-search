import sqlite3

import sqlite_vec
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.config import DATA_DIR, DATABASE_URL, EMBEDDING_DIMENSIONS, UPLOAD_DIR
from app.models import Base

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
    pool_pre_ping=True,
)


@event.listens_for(Engine, "connect")
def configure_sqlite(dbapi_connection: sqlite3.Connection, _connection_record: object) -> None:
    dbapi_connection.execute("PRAGMA foreign_keys = ON")
    dbapi_connection.execute("PRAGMA journal_mode = WAL")
    dbapi_connection.execute("PRAGMA busy_timeout = 30000")
    dbapi_connection.enable_load_extension(True)
    try:
        sqlite_vec.load(dbapi_connection)
    finally:
        dbapi_connection.enable_load_extension(False)


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create relational metadata and the sqlite-vec virtual table once at startup."""
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0("
                f"embedding float[{EMBEDDING_DIMENSIONS}], "
                "+chunk_id TEXT, "
                "file_id TEXT partition key)"
            )
        )
