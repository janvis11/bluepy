"""MCP (Model Context Protocol) translator for structured LLM outputs."""

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from groq import Groq

load_dotenv()


class QueryIntent(str, Enum):
    """Query intent types."""

    SELECT = "SELECT" 
    AGGREGATE = "AGGREGATE" 
    NEAREST = "NEAREST" 
    PLOT = "PLOT" 
    EXPLAIN = "EXPLAIN" 
    UNKNOWN = "UNKNOWN"


class VisualizationType(str, Enum):
    """Visualization types."""

    MAP = "map" 
    PROFILE = "profile" 
    TIMESERIES = "timeseries" 
    SCATTER = "scatter" 
    HEATMAP = "heatmap" 
    MULTI_PROFILE = "multi_profile" 
    NONE = "none" 


@dataclass
class MCPResponse:
    """Structured response from MCP translator."""

    intent: QueryIntent
    sql: Optional[str] = None
    visualization: VisualizationType = VisualizationType.NONE
    notes: str = ""
    confidence: float = 1.0
    clarification_needed: bool = False
    clarification_questions: List[str] = None
    parameters: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "intent": self.intent.value,
            "sql": self.sql,
            "visualization": self.visualization.value,
            "notes": self.notes,
            "confidence": self.confidence,
            "clarification_needed": self.clarification_needed,
            "clarification_questions": self.clarification_questions or [],
            "parameters": self.parameters or {},
        }


class MCPTranslator:
    """Translate natural language queries to structured SQL using MCP."""

    SCHEMA_INFO = """
    Database Schema:
    
    1. argo_profile_meta: Profile metadata with spatial indexing
       - profile_id (PRIMARY KEY)
       - float_wmo (VARCHAR)
       - cycle_number (INTEGER)
       - timestamp (TIMESTAMP)
       - latitude, longitude (FLOAT)
       - geom (GEOMETRY POINT)
       - max_depth (FLOAT)
       - variables (TEXT[])
       - summary (TEXT)
    
    2. argo_profile_jsonb: Profile measurements (fast retrieval)
       - profile_id (FOREIGN KEY)
       - depths (FLOAT[])
       - temperatures (FLOAT[])
       - salinities (FLOAT[])
       - dissolved_oxygens (FLOAT[])
       - chlorophylls (FLOAT[])
    
    3. argo_float: Float metadata
       - float_id (PRIMARY KEY)
       - wmo_number (VARCHAR)
       - platform_type (VARCHAR)
       - status (VARCHAR)
       - total_cycles (INTEGER)
    
    Spatial Functions:
    - ST_DWithin(geom, ST_SetSRID(ST_Point(lon, lat), 4326), radius_meters)
    - ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326)
    - ST_Distance(geom::geography, point::geography)
    """

    SQL_EXAMPLES = """
    Example Queries:
    
    1. Find profiles in a region:
    SELECT profile_id, float_wmo, timestamp, latitude, longitude
    FROM argo_profile_meta
    WHERE ST_Within(geom, ST_MakeEnvelope(40, -5, 80, 25, 4326))
    AND timestamp BETWEEN '2023-01-01' AND '2023-12-31';
    
    2. Get profile data with measurements:
    SELECT pm.profile_id, pm.float_wmo, pm.timestamp, pm.latitude, pm.longitude,
           pj.depths, pj.temperatures, pj.salinities
    FROM argo_profile_meta pm
    JOIN argo_profile_jsonb pj ON pm.profile_id = pj.profile_id
    WHERE pm.profile_id = 123;
    
    3. Find profiles near a point:
    SELECT profile_id, float_wmo, timestamp, latitude, longitude,
           ST_Distance(geom::geography, ST_SetSRID(ST_Point(45.5, 10.2), 4326)::geography) as distance_m
    FROM argo_profile_meta
    WHERE ST_DWithin(geom::geography, ST_SetSRID(ST_Point(45.5, 10.2), 4326)::geography, 500000)
    ORDER BY distance_m LIMIT 10;
    
    4. Aggregate statistics:
    SELECT AVG(unnest(temperatures)) as avg_temp, AVG(unnest(salinities)) as avg_sal
    FROM argo_profile_jsonb pj
    JOIN argo_profile_meta pm ON pj.profile_id = pm.profile_id
    WHERE pm.timestamp BETWEEN '2023-01-01' AND '2023-12-31'
    AND 'temperature' = ANY(pm.variables);
    """

    SYSTEM_PROMPT = f"""You are a database query translator for ARGO oceanographic data. Your task is to convert natural language queries into structured SQL queries and determine appropriate visualizations.

{SCHEMA_INFO}

{SQL_EXAMPLES}

CRITICAL INSTRUCTIONS:
1. Output ONLY valid JSON in this exact format:
{{
  "intent": "SELECT|AGGREGATE|NEAREST|PLOT|EXPLAIN|UNKNOWN",
  "sql": "valid SQL query or null",
  "visualization": "map|profile|timeseries|scatter|heatmap|multi_profile|none",
  "notes": "brief explanation",
  "confidence": 0.0-1.0,
  "clarification_needed": true/false,
  "clarification_questions": ["question1", "question2"],
  "parameters": {{"key": "value"}}
}}

2. SQL MUST be valid PostgreSQL with PostGIS
3. Use table aliases (pm, pj, af)
4. Always include necessary JOINs
5. Use ST_* functions for spatial queries
6. Handle array columns with unnest() or array operations
7. Add LIMIT clauses to prevent huge result sets
8. For profile plots, include depths and values arrays
9. Set confidence < 0.7 if query is ambiguous
10. Set clarification_needed=true if critical info is missing

VISUALIZATION RULES:
- "map": Geographic distribution of floats/profiles
- "profile": Depth vs variable (temperature, salinity, etc.)
- "multi_profile": Multiple profiles overlaid
- "timeseries": Variable over time
- "scatter": Two variables correlation
- "heatmap": Spatial or temporal heatmap
- "none": Just data retrieval

SAFETY:
- NO DROP, DELETE, UPDATE, INSERT statements
- NO system tables or pg_* queries
- Read-only SELECT queries only
"""

    def __init__(self, llm_provider: str = "groq", model: str = None):
        """
        Initialize MCP translator.

        Args:
            llm_provider: 'groq', 'openai' or 'anthropic'
            model: Model name
        """
        self.provider = llm_provider
        
        if llm_provider == "groq":
            self.model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        elif llm_provider == "openai":
            self.model = model or os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise NotImplementedError(f"Provider {llm_provider} not yet implemented")

        logger.info(f"Initialized MCP translator with {llm_provider}/{self.model}")

    def translate(
        self,
        user_query: str,
        retrieved_context: Optional[List[Dict]] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> MCPResponse:
        """
        Translate natural language query to structured SQL.

        Args:
            user_query: User's natural language query
            retrieved_context: Retrieved profile summaries from vector DB
            conversation_history: Previous conversation messages

        Returns:
            MCPResponse with SQL and metadata
        """
        context_text = self._build_context(retrieved_context)

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        if conversation_history:
            messages.extend(conversation_history[-4:]) 

        user_message = f"""User Query: {user_query}

Retrieved Context (relevant profiles):
{context_text}

Generate the structured JSON response."""

        messages.append({"role": "user", "content": user_message})

        try:
            if self.provider == "groq":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1, 
                    max_tokens=2000,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1, 
                    max_tokens=2000,
                    response_format={"type": "json_object"}, 
                )

            response_text = response.choices[0].message.content
            logger.debug(f"LLM response: {response_text}")

            mcp_response = self._parse_response(response_text)

            if mcp_response.sql:
                mcp_response.sql = self._validate_sql(mcp_response.sql)

            return mcp_response

        except Exception as e:
            logger.error(f"MCP translation error: {e}")
            return MCPResponse(
                intent=QueryIntent.UNKNOWN,
                notes=f"Error processing query: {str(e)}",
                confidence=0.0,
                clarification_needed=True,
                clarification_questions=["Could you rephrase your question?"],
            )

    def _build_context(self, retrieved_context: Optional[List[Dict]]) -> str:
        """Build context string from retrieved profiles."""
        if not retrieved_context:
            return "No specific profiles retrieved."

        context_parts = []
        for i, item in enumerate(retrieved_context[:5], 1): 
            context_parts.append(f"{i}. {item.get('summary', 'N/A')}")

        return "\n".join(context_parts)

    def _parse_response(self, response_text: str) -> MCPResponse:
        """Parse LLM JSON response into MCPResponse."""
        try:
            data = json.loads(response_text)

            return MCPResponse(
                intent=QueryIntent(data.get("intent", "UNKNOWN")),
                sql=data.get("sql"),
                visualization=VisualizationType(data.get("visualization", "none")),
                notes=data.get("notes", ""),
                confidence=float(data.get("confidence", 1.0)),
                clarification_needed=bool(data.get("clarification_needed", False)),
                clarification_questions=data.get("clarification_questions", []),
                parameters=data.get("parameters", {}),
            )
        except Exception as e:
            logger.error(f"Error parsing MCP response: {e}")
            return MCPResponse(
                intent=QueryIntent.UNKNOWN,
                notes=f"Failed to parse response: {str(e)}",
                confidence=0.0,
            )

    def _validate_sql(self, sql: str) -> str:
        """
        Validate and sanitize SQL query.

        Args:
            sql: SQL query string

        Returns:
            Validated SQL or raises exception
        """
        sql = sql.strip()

        sql = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)

        dangerous_keywords = [
            "DROP",
            "DELETE",
            "UPDATE",
            "INSERT",
            "ALTER",
            "CREATE",
            "TRUNCATE",
            "GRANT",
            "REVOKE",
            "EXECUTE",
        ]

        sql_upper = sql.upper()
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                raise ValueError(f"Dangerous SQL keyword detected: {keyword}")

        if not sql_upper.strip().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")

        if "LIMIT" not in sql_upper:
            sql += " LIMIT 10000"

        return sql

    def generate_explanation(self, query_result: Dict, user_query: str) -> str:
        """
        Generate natural language explanation of query results.

        Args:
            query_result: Query execution results
            user_query: Original user query

        Returns:
            Natural language explanation
        """
        messages = [
            {
                "role": "system",
                "content": "You are an oceanography data assistant. Explain query results in clear, concise language.",
            },
            {
                "role": "user",
                "content": f"""User asked: "{user_query}"

Query returned {query_result.get('row_count', 0)} results.

Sample data: {json.dumps(query_result.get('sample_rows', [])[:3], indent=2)}

Provide a brief, informative explanation of these results.""",
            },
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return f"Found {query_result.get('row_count', 0)} results."
