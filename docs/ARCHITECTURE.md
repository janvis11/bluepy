# BluePy Architecture Documentation

## System Overview

BluePy is an AI-powered conversational interface for ARGO oceanographic data, implementing a complete RAG (Retrieval-Augmented Generation) pipeline with MCP (Model Context Protocol) for structured LLM outputs.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Streamlit Frontend (Port 8501)                          │  │
│  │  - Chat Interface                                        │  │
│  │  - Interactive Maps (Folium/Leaflet)                     │  │
│  │  - Profile Visualizations (Plotly)                       │  │
│  │  - Filters & Controls                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP/REST
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (Port 8000)                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  API Endpoints                                           │  │
│  │  - /chat (POST)         - /profiles (GET)                │  │
│  │  - /profile/{id} (GET)  - /map/geojson (GET)            │  │
│  │  - /statistics (GET)    - /health (GET)                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  RAG Orchestrator                                        │  │
│  │  ┌────────────┐  ┌────────────┐  ┌─────────────────┐   │  │
│  │  │ Retriever  │→ │    MCP     │→ │  SQL Executor   │   │  │
│  │  │ (Vector DB)│  │ Translator │  │  (PostgreSQL)   │   │  │
│  │  └────────────┘  └────────────┘  └─────────────────┘   │  │
│  │         ↓              ↓                  ↓              │  │
│  │  Context Retrieval → SQL Generation → Query Results     │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────────┐  ┌─────────────────┐
│ PostgreSQL   │  │   ChromaDB       │  │  OpenAI API     │
│ + PostGIS    │  │ (Vector Store)   │  │  (LLM/Embed)    │
│              │  │                  │  │                 │
│ - Profiles   │  │ - Embeddings     │  │ - GPT-4         │
│ - Metadata   │  │ - Summaries      │  │ - Embeddings    │
│ - Spatial    │  │ - Documents      │  │                 │
└──────────────┘  └──────────────────┘  └─────────────────┘
```

## Data Flow

### 1. Ingestion Pipeline

```
NetCDF Files → Parser → Normalizer → Dual Storage
                                      ├→ PostgreSQL (structured queries)
                                      └→ Parquet (analytics/archival)
```

**Components:**
- `ingestion/parsers/netcdf_parser.py`: Parse ARGO NetCDF files
- `ingestion/pipeline.py`: Orchestrate ingestion process
- Outputs: PostgreSQL tables + Parquet files

### 2. Embedding Generation

```
Profile Summaries → Embedding Model → Vector DB
                    (OpenAI/Local)    (ChromaDB)
```

**Components:**
- `backend/rag/embeddings.py`: Generate embeddings
- `scripts/embed_profiles.py`: Batch embedding script
- Storage: ChromaDB persistent storage

### 3. Query Processing (RAG + MCP)

```
User Query → Embedding → Vector Search → Context Retrieval
                                              ↓
                                         MCP Translator
                                         (LLM + Prompt)
                                              ↓
                                    Structured JSON Output
                                    {intent, sql, viz, ...}
                                              ↓
                                      SQL Validation
                                              ↓
                                      Query Execution
                                              ↓
                                    Results + Explanation
```

**Components:**
- `backend/rag/rag_orchestrator.py`: Main orchestration
- `backend/rag/mcp_translator.py`: LLM-based SQL generation
- `backend/rag/embeddings.py`: Semantic search

## Database Schema

### Core Tables

**argo_float**
- Float metadata (WMO number, platform type, status)
- One-to-many relationship with profiles

**argo_profile_meta**
- Profile metadata with spatial indexing (PostGIS)
- Includes: location, timestamp, depth, variables, summary
- Indexed: geom (GIST), timestamp, float_wmo

**argo_profile_jsonb**
- Profile measurements stored as arrays (JSONB)
- Fast retrieval for visualization
- Arrays: depths, temperatures, salinities, etc.

**argo_profile_data** (optional)
- Normalized format (one row per depth level)
- For complex analytical queries

**query_history**
- Tracks user queries for analytics and caching

**embeddings_metadata**
- Links to vector DB embeddings

### Spatial Functions

- `get_profiles_in_bbox()`: Bounding box queries
- `get_profiles_near_point()`: Radius-based search
- PostGIS functions: `ST_Within`, `ST_DWithin`, `ST_Distance`

## MCP (Model Context Protocol)

### Purpose
Ensure LLM outputs are structured, validated, and safe.

### MCP Response Format
```json
{
  "intent": "SELECT|AGGREGATE|NEAREST|PLOT|EXPLAIN",
  "sql": "SELECT ... FROM ... WHERE ...",
  "visualization": "map|profile|timeseries|scatter|none",
  "notes": "Brief explanation",
  "confidence": 0.0-1.0,
  "clarification_needed": false,
  "clarification_questions": [],
  "parameters": {}
}
```

### Safety Features
- SQL validation (whitelist SELECT only)
- Dangerous keyword detection (DROP, DELETE, etc.)
- Automatic LIMIT injection
- Read-only database user (recommended)

## API Endpoints

### Chat & Query
- `POST /chat`: Main conversational interface
- `GET /statistics`: Database statistics

### Data Access
- `GET /profiles`: List profiles with filters
- `GET /profile/{id}`: Detailed profile data
- `GET /floats`: List floats
- `GET /map/geojson`: Geographic data for mapping

### Session Management
- `DELETE /session/{id}`: Clear conversation history

## Frontend Architecture

### Streamlit Components

**Tab 1: Chat Interface**
- Natural language input
- Streaming responses
- SQL display (expandable)
- Results as DataFrame

**Tab 2: Map Explorer**
- Folium/Leaflet interactive map
- Profile markers with popups
- Filter controls

**Tab 3: Profile Viewer**
- Plotly depth profiles
- Variable selection
- Raw data table

### State Management
- `session_id`: Conversation tracking
- `chat_history`: Message history
- `current_results`: Latest query results
- `selected_profile`: Active profile for visualization

## Deployment Options

### Local Development
```bash
# Terminal 1: Backend
uvicorn backend.main:app --reload

# Terminal 2: Frontend
streamlit run frontend/app.py
```

### Docker Compose
```bash
docker-compose up -d
```

Services:
- PostgreSQL + PostGIS
- Redis (caching)
- Backend API
- Frontend UI

### Production Considerations
- Use environment-specific `.env` files
- Enable HTTPS/TLS
- Set up reverse proxy (nginx)
- Configure database connection pooling
- Implement rate limiting
- Add monitoring (Prometheus/Grafana)
- Set up log aggregation

## Performance Optimization

### Database
- Spatial indexes (GIST) on geometry columns
- B-tree indexes on timestamp, float_wmo
- Materialized views for statistics
- Connection pooling (SQLAlchemy)

### Vector DB
- Batch embedding generation
- HNSW index for fast similarity search
- Persistent storage to avoid re-indexing

### Caching
- Redis for query results
- Session-based context caching
- Embedding cache for repeated queries

### API
- Async endpoints where applicable
- Response pagination
- Streaming for large results

## Security

### API Security
- CORS configuration
- Input validation (Pydantic)
- SQL injection prevention (parameterized queries)
- Rate limiting (recommended)

### Database Security
- Read-only user for query execution
- SQL whitelist enforcement
- Prepared statements

### Secrets Management
- Environment variables for API keys
- Never commit `.env` files
- Use secret management services in production

## Monitoring & Logging

### Logging
- Loguru for structured logging
- Log levels: DEBUG, INFO, WARNING, ERROR
- Log rotation and retention

### Metrics
- Query execution time
- API response time
- Database connection pool stats
- Vector DB query performance

### Health Checks
- `/health` endpoint
- Database connectivity
- RAG orchestrator status

## Extension Points

### Adding New Variables
1. Update `netcdf_parser.py` VAR_MAPPINGS
2. Add columns to `argo_profile_jsonb`
3. Update frontend variable selection

### Custom Visualizations
1. Add plot function in `frontend/app.py`
2. Update MCP visualization types
3. Add corresponding API endpoint if needed

### Alternative LLM Providers
1. Implement provider in `mcp_translator.py`
2. Add configuration in `.env`
3. Update initialization logic

### Additional Data Sources
1. Create new parser in `ingestion/parsers/`
2. Update pipeline to handle new format
3. Extend database schema if needed

## Testing Strategy

### Unit Tests
- Parser logic
- Embedding generation
- MCP translation
- SQL validation

### Integration Tests
- End-to-end query processing
- Database operations
- API endpoints

### Performance Tests
- Large dataset ingestion
- Concurrent query handling
- Vector search scalability

## Troubleshooting

### Common Issues

**Database Connection**
- Check PostgreSQL is running
- Verify DATABASE_URL
- Ensure PostGIS extension installed

**Embedding Generation**
- Verify API keys
- Check rate limits
- Consider local models for development

**Frontend Not Loading Data**
- Check API_BASE_URL
- Verify backend is running
- Check browser console for errors

**Slow Queries**
- Add database indexes
- Optimize SQL queries
- Enable query caching

## Future Enhancements

- Multi-user authentication
- Real-time data ingestion
- Advanced analytics dashboard
- Export to multiple formats
- Mobile-responsive UI
- Collaborative features
- Custom alert system
- Integration with other ocean data sources
