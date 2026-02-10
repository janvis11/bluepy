"""Embeddings generation and vector database management."""

import hashlib
import os
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()


class EmbeddingGenerator:
    """Generate embeddings using OpenAI or local models."""

    def __init__(self, provider: str = "openai", model: str = None):
        """
        Initialize embedding generator.

        Args:
            provider: 'openai' or 'local'
            model: Model name (e.g., 'text-embedding-3-small' for OpenAI)
        """
        self.provider = provider
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        if provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.dimension = 1536 
        else:
            self.model = model or "all-MiniLM-L6-v2"
            self.encoder = SentenceTransformer(self.model)
            self.dimension = self.encoder.get_sentence_embedding_dimension()

        logger.info(f"Initialized {provider} embeddings with model {self.model} (dim={self.dimension})")

    def generate(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.generate_batch([text])[0]

    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        if self.provider == "openai":
            return self._generate_openai(texts)
        else:
            return self._generate_local(texts)

    def _generate_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            response = self.client.embeddings.create(input=texts, model=self.model)
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise

    def _generate_local(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model."""
        try:
            embeddings = self.encoder.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Local embedding error: {e}")
            raise


class VectorDatabase:
    """Vector database interface using ChromaDB."""

    def __init__(self, persist_dir: str = None, collection_name: str = "argo_profiles"):
        """
        Initialize vector database.

        Args:
            persist_dir: Directory to persist database
            collection_name: Name of the collection
        """
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/embeddings/chroma")
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}, 
        )

        logger.info(f"Initialized ChromaDB: {persist_dir}, collection={collection_name}")

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
    ):
        """
        Add embeddings to the database.

        Args:
            ids: Unique IDs for each embedding
            embeddings: Embedding vectors
            documents: Original text documents
            metadatas: Optional metadata dictionaries
        """
        try:
            self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
            logger.debug(f"Added {len(ids)} embeddings to {self.collection_name}")
        except Exception as e:
            logger.error(f"Error adding to vector DB: {e}")
            raise

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ) -> Dict:
        """
        Query the vector database.

        Args:
            query_embeddings: Query embedding vectors
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document content filter

        Returns:
            Query results with ids, documents, metadatas, distances
        """
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document,
            )
            return results
        except Exception as e:
            logger.error(f"Error querying vector DB: {e}")
            raise

    def delete(self, ids: List[str]):
        """Delete embeddings by IDs."""
        try:
            self.collection.delete(ids=ids)
            logger.debug(f"Deleted {len(ids)} embeddings from {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting from vector DB: {e}")
            raise

    def count(self) -> int:
        """Get count of embeddings in collection."""
        return self.collection.count()

    def reset(self):
        """Reset the collection (delete all data)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"Reset collection {self.collection_name}")


class ProfileEmbedder:
    """Embed ARGO profile summaries and metadata."""

    def __init__(self, embedding_generator: EmbeddingGenerator, vector_db: VectorDatabase):
        """
        Initialize profile embedder.

        Args:
            embedding_generator: Embedding generator instance
            vector_db: Vector database instance
        """
        self.embedding_gen = embedding_generator
        self.vector_db = vector_db

    def embed_profile_summaries(self, profiles: List[Dict]) -> List[str]:
        """
        Embed profile summaries and store in vector DB.

        Args:
            profiles: List of profile dictionaries with 'profile_id', 'summary', metadata

        Returns:
            List of embedding IDs
        """
        if not profiles:
            return []

        summaries = [p["summary"] for p in profiles]
        profile_ids = [str(p["profile_id"]) for p in profiles]

        logger.info(f"Generating embeddings for {len(summaries)} profiles...")
        embeddings = self.embedding_gen.generate_batch(summaries)

        embedding_ids = [self._generate_embedding_id(pid) for pid in profile_ids]

        metadatas = []
        for p in profiles:
            metadata = {
                "profile_id": p["profile_id"],
                "float_wmo": p.get("float_wmo", ""),
                "cycle_number": p.get("cycle_number", 0),
                "timestamp": p.get("timestamp", ""),
                "latitude": p.get("latitude", 0.0),
                "longitude": p.get("longitude", 0.0),
                "content_type": "profile_summary",
            }
            metadatas.append(metadata)

        self.vector_db.add(ids=embedding_ids, embeddings=embeddings, documents=summaries, metadatas=metadatas)

        logger.info(f"Embedded {len(profiles)} profiles")
        return embedding_ids

    def embed_documentation(self, docs: List[Dict]) -> List[str]:
        """
        Embed documentation and domain knowledge.

        Args:
            docs: List of documentation dictionaries with 'text', 'title', 'category'

        Returns:
            List of embedding IDs
        """
        if not docs:
            return []

        texts = [d["text"] for d in docs]
        embeddings = self.embedding_gen.generate_batch(texts)

        embedding_ids = [self._generate_embedding_id(f"doc_{i}") for i in range(len(docs))]

        metadatas = []
        for d in docs:
            metadata = {
                "title": d.get("title", ""),
                "category": d.get("category", "documentation"),
                "content_type": "documentation",
            }
            metadatas.append(metadata)

        self.vector_db.add(ids=embedding_ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

        logger.info(f"Embedded {len(docs)} documentation items")
        return embedding_ids

    def search_profiles(
        self, query: str, n_results: int = 10, filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for relevant profiles using natural language query.

        Args:
            query: Natural language query
            n_results: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of relevant profile metadata
        """
        query_embedding = self.embedding_gen.generate(query)

        where_filter = {"content_type": "profile_summary"}
        if filters:
            where_filter.update(filters)

        results = self.vector_db.query(query_embeddings=[query_embedding], n_results=n_results, where=where_filter)

        formatted_results = []
        if results["ids"] and len(results["ids"]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                result = {
                    "embedding_id": doc_id,
                    "profile_id": results["metadatas"][0][i].get("profile_id"),
                    "summary": results["documents"][0][i],
                    "distance": results["distances"][0][i],
                    "metadata": results["metadatas"][0][i],
                }
                formatted_results.append(result)

        return formatted_results

    def search_documentation(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant documentation."""
        query_embedding = self.embedding_gen.generate(query)

        results = self.vector_db.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"content_type": "documentation"},
        )

        formatted_results = []
        if results["ids"] and len(results["ids"]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                result = {
                    "embedding_id": doc_id,
                    "text": results["documents"][0][i],
                    "distance": results["distances"][0][i],
                    "metadata": results["metadatas"][0][i],
                }
                formatted_results.append(result)

        return formatted_results

    def _generate_embedding_id(self, identifier: str) -> str:
        """Generate unique embedding ID."""
        return hashlib.md5(identifier.encode()).hexdigest()
