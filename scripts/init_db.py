"""Initialize database schema and create tables."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from backend.db.database import init_db, test_connection


def main():
    """Initialize database."""
    logger.info("Starting database initialization...")

    if not test_connection():
        logger.error("Database connection failed. Please check your configuration.")
        return 1

    try:
        init_db()
        logger.success("✓ Database initialized successfully!")
        return 0
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
