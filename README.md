# BluePy - AI Conversational Interface for ARGO Data

An intelligent conversational interface for querying and visualizing ARGO oceanographic float data using RAG (Retrieval-Augmented Generation) and MCP (Model Context Protocol).

## Architecture

**Data Flow:** Ingest Argo NetCDF → normalize & store (Postgres + Parquet) → index metadata & embeddings (FAISS/Chroma) → RAG + MCP translator (LLM) → Backend APIs → Interactive dashboard + Chat UI (Streamlit) + visualizations (Plotly/Leaflet)
![Alt text for accessibility](bluepy_rag_arch.png)

## Features

- **ARGO Data Ingestion**: Parse NetCDF files and normalize to structured formats
- **Dual Storage**: PostgreSQL with PostGIS for spatial queries + Parquet for analytics
- **Vector Search**: FAISS/Chroma for semantic retrieval of profiles and metadata
- **RAG + MCP**: LLM-powered natural language to SQL translation with structured outputs
- **FastAPI Backend**: RESTful APIs for chat, queries, and data access
- **Interactive Frontend**: Streamlit dashboard with chat, maps, and visualizations
- **Geospatial Viz**: Leaflet maps for float trajectories, Plotly for profiles

## Usage Examples

### Natural Language Queries

- "Show me salinity profiles near the equator in March 2023"
- "What's the average temperature at 500m depth in the Indian Ocean?"
- "Find floats with anomalous oxygen levels in the last 6 months"
- "Plot temperature vs depth for float 2902123"

## Technology Stack

- **Backend**: FastAPI, SQLAlchemy, psycopg2
- **Database**: PostgreSQL + PostGIS, Parquet (PyArrow)
- **Vector DB**: ChromaDB / FAISS
- **LLM**: OpenAI GPT-4 / Anthropic Claude
- **Frontend**: Streamlit, Plotly, Folium/Leaflet
- **Data Processing**: xarray, netCDF4, pandas, numpy
- **Deployment**: Docker, Docker Compose, Kubernetes