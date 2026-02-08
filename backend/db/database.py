"""Database connection and session management."""

import os
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from backend.db.models import Base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://argo_user:argo_password@localhost:5432/argo_db")

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True, 
    echo=False, 
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@event.listens_for(Engine, "connect")
def set_postgis(dbapi_conn, connection_record):
    """Ensure PostGIS is available."""
    cursor = dbapi_conn.cursor()
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        dbapi_conn.commit()
    except Exception:
        dbapi_conn.rollback()
    finally:
        cursor.close()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database session.
    
    Usage:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database session.
    
    Usage:
        with get_db_context() as db:
            items = db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")


def drop_db():
    """Drop all database tables. Use with caution!"""
    Base.metadata.drop_all(bind=engine)
    print("Database tables dropped.")


def reset_db():
    """Reset database (drop and recreate all tables)."""
    drop_db()
    init_db()
    print("Database reset complete.")


def test_connection():
    """Test database connection."""
    try:
        with engine.connect() as conn:
            result = conn.execute("SELECT version();")
            version = result.fetchone()[0]
            print(f"✓ Database connection successful!")
            print(f"PostgreSQL version: {version}")
            
            result = conn.execute("SELECT PostGIS_version();")
            postgis_version = result.fetchone()[0]
            print(f"PostGIS version: {postgis_version}")
            return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


if __name__ == "__main__":
    if test_connection():
        init_db()
