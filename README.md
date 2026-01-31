# BluePy - AI Conversational Interface for ARGO Data

An intelligent conversational interface for querying and visualizing ARGO oceanographic float data using RAG (Retrieval-Augmented Generation) and MCP (Model Context Protocol).

## Architecture

**Data Flow:** Ingest Argo NetCDF → normalize & store (Postgres + Parquet) → index metadata & embeddings (FAISS/Chroma) → RAG + MCP translator (LLM) → Backend APIs → Interactive dashboard + Chat UI (Streamlit) + visualizations (Plotly/Leaflet)
![Alt text for accessibility](bluepy_rag_arch.png)

## Features

- 🌊 **ARGO Data Ingestion**: Parse NetCDF files and normalize to structured formats
- 🗄️ **Dual Storage**: PostgreSQL with PostGIS for spatial queries + Parquet for analytics
- 🔍 **Vector Search**: FAISS/Chroma for semantic retrieval of profiles and metadata
- 🤖 **RAG + MCP**: LLM-powered natural language to SQL translation with structured outputs
- 🚀 **FastAPI Backend**: RESTful APIs for chat, queries, and data access
- 📊 **Interactive Frontend**: Streamlit dashboard with chat, maps, and visualizations
- 🗺️ **Geospatial Viz**: Leaflet maps for float trajectories, Plotly for profiles

## Project Structure

```
bluepy/
├── backend/
│   ├── db/               # Database models, schema, connections
│   ├── rag/              # RAG pipeline + MCP translator
│   ├── main.py           # FastAPI app entry point
│   └── __init__.py
├── frontend/
│   ├── app.py            # Streamlit main app (Premium UI)
│   ├── styles.py         # Custom CSS (Glass morphism theme)
│   └── __pycache__/
├── ingestion/
│   ├── parsers/          # NetCDF file parsers
│   ├── pipeline.py       # Data ingestion orchestration
│   └── __init__.py
├── scripts/
│   ├── sample_data_generator.py  # Generate realistic ARGO data
│   ├── init_db.py                # Database initialization
│   ├── embed_profiles.py         # Generate embeddings
│   ├── run_all.bat/sh            # Full stack launcher
│   └── ...
├── data/
│   ├── raw/              # Raw NetCDF files
│   ├── processed/        # Parquet files
│   └── embeddings/       # ChromaDB vector storage
├── docs/                 # 📚 Complete documentation
│   ├── ARCHITECTURE.md
│   ├── API_DOCUMENTATION.md
│   ├── QUICKSTART.md
│   ├── DEPLOYMENT.md
│   └── ... (12 files total)
├── tests/                # Unit & integration tests
├── docker/               # Dockerfile configurations
├── setup.bat             # Complete setup script
├── demo_frontend.bat     # Quick UI demo launcher
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ with PostGIS extension
- Docker (optional, for containerized deployment)

### Installation

1. Clone and navigate to the project:
```bash
cd bluepy
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Groq API key and database credentials
```

5. Generate sample data (375 realistic ARGO profiles):
```bash
python scripts/sample_data_generator.py --num-floats 15 --region all
```

6. Initialize database:
```bash
python scripts/init_db.py
```

6. Run data ingestion (example):
```bash
python ingestion/pipeline.py --input data/raw --output data/processed
```

### Running the Application

**Backend API:**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Frontend Dashboard:**
```bash
streamlit run frontend/app.py --server.port 8501
```

Access the application at `http://localhost:8501`

## Usage Examples

### Natural Language Queries

- "Show me salinity profiles near the equator in March 2023"
- "What's the average temperature at 500m depth in the Indian Ocean?"
- "Find floats with anomalous oxygen levels in the last 6 months"
- "Plot temperature vs depth for float 2902123"

### API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /chat` - AI conversational interface (RAG + MCP)
- `GET /profiles` - List profiles with filters
- `GET /profile/{profile_id}` - Get specific profile details
- `GET /floats` - List all floats with filters
- `GET /statistics` - Database statistics
- `GET /map/geojson` - GeoJSON data for mapping
- `DELETE /session/{session_id}` - Clear conversation session

## Configuration

Key environment variables in `.env`:

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/argo_db
POSTGRES_USER=argo_user
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=argo_db

# LLM Configuration (Using Groq for fast inference)
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector DB
VECTOR_DB_TYPE=chroma  # or faiss
CHROMA_PERSIST_DIR=./data/embeddings/chroma

# API
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:8501

# Frontend
STREAMLIT_SERVER_PORT=8501
MAP_PROVIDER=leaflet
```

## Development

### Running Tests
```bash
pytest tests/ -v --cov=backend --cov=ingestion
```

### Code Quality
```bash
# Linting
flake8 backend/ ingestion/ frontend/

# Type checking
mypy backend/ ingestion/

# Formatting
black backend/ ingestion/ frontend/
```

## Deployment

### Docker Compose
```bash
docker-compose up -d
```

### Kubernetes
```bash
kubectl apply -f k8s/
```

## Data Schema

### PostgreSQL Tables

**argo_profile** - Main profile data table with spatial indexing
**argo_profile_meta** - Profile metadata and summaries
**argo_float** - Float information and trajectories

See `backend/db/schema.sql` for complete schema definitions.

## Technology Stack

- **Backend**: FastAPI, SQLAlchemy, psycopg2
- **Database**: PostgreSQL + PostGIS, Parquet (PyArrow)
- **Vector DB**: ChromaDB / FAISS
- **LLM**: OpenAI GPT-4 / Anthropic Claude
- **Frontend**: Streamlit, Plotly, Folium/Leaflet
- **Data Processing**: xarray, netCDF4, pandas, numpy
- **Deployment**: Docker, Docker Compose, Kubernetes

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- ARGO Program for oceanographic data
- OpenAI for LLM capabilities
- Streamlit community for excellent framework

## Support

For issues and questions, please open a GitHub issue or contact the maintainers.
