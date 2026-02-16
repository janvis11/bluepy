"""Generate embeddings for existing profiles in database."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from sqlalchemy import text

from backend.db.database import get_db_context
from backend.rag.embeddings import EmbeddingGenerator, ProfileEmbedder, VectorDatabase


def main():
    """Generate embeddings for profiles."""
    parser = argparse.ArgumentParser(description="Generate embeddings for ARGO profiles")
    parser.add_argument("--provider", default="openai", choices=["openai", "local"], help="Embedding provider")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of profiles to process")
    args = parser.parse_args()

    logger.info(f"Starting embedding generation (provider={args.provider})")

    embedding_gen = EmbeddingGenerator(provider=args.provider)
    vector_db = VectorDatabase()
    embedder = ProfileEmbedder(embedding_gen, vector_db)

    with get_db_context() as db:
        query = """
        SELECT profile_id, float_wmo, cycle_number, timestamp, 
               latitude, longitude, summary
        FROM argo_profile_meta
        WHERE summary IS NOT NULL
        """
        if args.limit:
            query += f" LIMIT {args.limit}"

        result = db.execute(text(query))
        rows = result.fetchall()

        logger.info(f"Found {len(rows)} profiles to embed")

        for i in range(0, len(rows), args.batch_size):
            batch = rows[i : i + args.batch_size]

            profiles = []
            for row in batch:
                profiles.append(
                    {
                        "profile_id": row[0],
                        "float_wmo": row[1],
                        "cycle_number": row[2],
                        "timestamp": str(row[3]),
                        "latitude": row[4],
                        "longitude": row[5],
                        "summary": row[6],
                    }
                )

            try:
                embedding_ids = embedder.embed_profile_summaries(profiles)
                logger.info(f"Batch {i
            except Exception as e:
                logger.error(f"Error embedding batch: {e}")

    logger.success(f"✓ Embedding generation complete! Total embeddings: {vector_db.count()}")
    return 0


if __name__ == "__main__":
    exit(main())
