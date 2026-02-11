"""RAG orchestrator combining retrieval, MCP translation, and query execution."""

import time
from typing import Dict, List, Optional

from loguru import logger
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.db.database import get_db_context
from backend.rag.embeddings import EmbeddingGenerator, ProfileEmbedder, VectorDatabase
from backend.rag.mcp_translator import MCPResponse, MCPTranslator, QueryIntent, VisualizationType


class RAGOrchestrator:
    """Orchestrate RAG pipeline: retrieval → MCP → execution → response."""

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_db: VectorDatabase,
        mcp_translator: MCPTranslator,
    ):
        """
        Initialize RAG orchestrator.

        Args:
            embedding_generator: Embedding generator instance
            vector_db: Vector database instance
            mcp_translator: MCP translator instance
        """
        self.embedder = ProfileEmbedder(embedding_generator, vector_db)
        self.mcp = mcp_translator
        self.conversation_history: Dict[str, List[Dict]] = {} 

    def process_query(
        self,
        user_query: str,
        session_id: str = "default",
        retrieve_context: bool = True,
        n_context: int = 10,
    ) -> Dict:
        """
        Process a user query through the RAG pipeline.

        Args:
            user_query: Natural language query
            session_id: Session identifier for conversation history
            retrieve_context: Whether to retrieve context from vector DB
            n_context: Number of context items to retrieve

        Returns:
            Response dictionary with SQL, results, visualization info
        """
        start_time = time.time()
        logger.info(f"Processing query (session={session_id}): {user_query}")

        retrieved_context = []
        if retrieve_context:
            try:
                retrieved_context = self.embedder.search_profiles(user_query, n_results=n_context)
                logger.info(f"Retrieved {len(retrieved_context)} relevant profiles")
            except Exception as e:
                logger.warning(f"Context retrieval failed: {e}")

        conversation_history = self.conversation_history.get(session_id, [])

        try:
            mcp_response = self.mcp.translate(
                user_query=user_query,
                retrieved_context=retrieved_context,
                conversation_history=conversation_history,
            )
            logger.info(f"MCP translation: intent={mcp_response.intent}, viz={mcp_response.visualization}")
        except Exception as e:
            logger.error(f"MCP translation failed: {e}")
            return self._error_response(f"Translation error: {str(e)}")

        if mcp_response.clarification_needed:
            return self._clarification_response(mcp_response)

        query_results = None
        if mcp_response.sql and mcp_response.intent != QueryIntent.EXPLAIN:
            try:
                query_results = self._execute_sql(mcp_response.sql)
                logger.info(f"Query executed: {query_results['row_count']} rows returned")
            except Exception as e:
                logger.error(f"SQL execution failed: {e}")
                return self._error_response(f"Query execution error: {str(e)}")

        explanation = ""
        if query_results:
            try:
                explanation = self.mcp.generate_explanation(query_results, user_query)
            except Exception as e:
                logger.warning(f"Explanation generation failed: {e}")
                explanation = f"Found {query_results['row_count']} results."

        self._update_conversation_history(session_id, user_query, mcp_response, explanation)

        elapsed_time = time.time() - start_time
        response = {
            "success": True,
            "query": user_query,
            "intent": mcp_response.intent.value,
            "sql": mcp_response.sql,
            "visualization": mcp_response.visualization.value,
            "explanation": explanation,
            "notes": mcp_response.notes,
            "confidence": mcp_response.confidence,
            "results": query_results,
            "context": retrieved_context[:3] if retrieved_context else [], 
            "elapsed_time_ms": int(elapsed_time * 1000),
        }

        logger.info(f"Query processed successfully in {elapsed_time:.2f}s")
        return response

    def _execute_sql(self, sql: str) -> Dict:
        """
        Execute SQL query and return results.

        Args:
            sql: SQL query string

        Returns:
            Dictionary with results, row_count, columns
        """
        with get_db_context() as db:
            result = db.execute(text(sql))
            rows = result.fetchall()
            columns = result.keys()

            results_list = []
            for row in rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    value = row[i]
                    if hasattr(value, "__geo_interface__"): 
                        value = str(value)
                    elif isinstance(value, (list, tuple)): 
                        value = list(value)
                    row_dict[col] = value
                results_list.append(row_dict)

            return {
                "rows": results_list,
                "row_count": len(results_list),
                "columns": list(columns),
                "sample_rows": results_list[:5], 
            }

    def _update_conversation_history(
        self, session_id: str, user_query: str, mcp_response: MCPResponse, explanation: str
    ):
        """Update conversation history for session."""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []

        self.conversation_history[session_id].append({"role": "user", "content": user_query})

        assistant_message = f"{explanation}\n\n[SQL: {mcp_response.sql}]" if mcp_response.sql else explanation
        self.conversation_history[session_id].append({"role": "assistant", "content": assistant_message})

        if len(self.conversation_history[session_id]) > 10:
            self.conversation_history[session_id] = self.conversation_history[session_id][-10:]

    def _clarification_response(self, mcp_response: MCPResponse) -> Dict:
        """Build clarification response."""
        return {
            "success": True,
            "clarification_needed": True,
            "questions": mcp_response.clarification_questions,
            "notes": mcp_response.notes,
            "confidence": mcp_response.confidence,
        }

    def _error_response(self, error_message: str) -> Dict:
        """Build error response."""
        return {
            "success": False,
            "error": error_message,
            "explanation": "I encountered an error processing your query. Please try rephrasing.",
        }

    def clear_session(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            logger.info(f"Cleared session: {session_id}")

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        with get_db_context() as db:
            stats_query = """
            SELECT 
                COUNT(DISTINCT float_wmo) as total_floats,
                COUNT(*) as total_profiles,
                MIN(timestamp) as earliest_date,
                MAX(timestamp) as latest_date,
                AVG(max_depth) as avg_max_depth
            FROM argo_profile_meta;
            """
            result = db.execute(text(stats_query))
            row = result.fetchone()

            return {
                "total_floats": row[0] if row else 0,
                "total_profiles": row[1] if row else 0,
                "earliest_date": str(row[2]) if row and row[2] else None,
                "latest_date": str(row[3]) if row and row[3] else None,
                "avg_max_depth": float(row[4]) if row and row[4] else 0.0,
            }


_orchestrator_instance: Optional[RAGOrchestrator] = None


def get_orchestrator() -> RAGOrchestrator:
    """Get global RAG orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        raise RuntimeError("RAG orchestrator not initialized. Call initialize_orchestrator() first.")
    return _orchestrator_instance


def initialize_orchestrator(
    embedding_provider: str = "local",
    llm_provider: str = "groq",
    vector_db_path: str = None,
):
    """
    Initialize global RAG orchestrator.

    Args:
        embedding_provider: 'openai' or 'local'
        llm_provider: 'groq', 'openai' or 'anthropic'
        vector_db_path: Path to vector database
    """
    global _orchestrator_instance

    logger.info("Initializing RAG orchestrator...")

    embedding_gen = EmbeddingGenerator(provider=embedding_provider)
    vector_db = VectorDatabase(persist_dir=vector_db_path)
    mcp_translator = MCPTranslator(llm_provider=llm_provider)

    _orchestrator_instance = RAGOrchestrator(
        embedding_generator=embedding_gen,
        vector_db=vector_db,
        mcp_translator=mcp_translator,
    )

    logger.info("RAG orchestrator initialized successfully")
    return _orchestrator_instance
