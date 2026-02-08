"""FastAPI main application."""

import os
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.db.database import get_db, test_connection
from backend.db.models import ArgoFloat, ArgoProfileMeta
from backend.rag.rag_orchestrator import get_orchestrator, initialize_orchestrator

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting BluePy API...")
    
    if not test_connection():
        logger.error("Database connection failed!")
    
    try:
        initialize_orchestrator(
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local"),
            llm_provider=os.getenv("LLM_PROVIDER", "groq"),
            vector_db_path=os.getenv("CHROMA_PERSIST_DIR"),
        )
        logger.info("RAG orchestrator initialized")
    except Exception as e:
        logger.error(f"Failed to initialize RAG orchestrator: {e}")
    
    yield
    
    logger.info("Shutting down BluePy API...")


app = FastAPI(
    title="BluePy API",
    description="AI Conversational Interface for ARGO Data",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8501").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    retrieve_context: bool = Field(True, description="Whether to retrieve context from vector DB")
    n_context: int = Field(10, description="Number of context items to retrieve", ge=1, le=50)


class ChatResponse(BaseModel):
    """Chat response model."""

    success: bool
    query: Optional[str] = None
    intent: Optional[str] = None
    sql: Optional[str] = None
    visualization: Optional[str] = None
    explanation: Optional[str] = None
    notes: Optional[str] = None
    confidence: Optional[float] = None
    results: Optional[Dict] = None
    context: Optional[List[Dict]] = None
    elapsed_time_ms: Optional[int] = None
    clarification_needed: Optional[bool] = None
    questions: Optional[List[str]] = None
    error: Optional[str] = None


class ProfileResponse(BaseModel):
    """Profile response model."""

    profile_id: int
    float_wmo: str
    cycle_number: int
    timestamp: str
    latitude: float
    longitude: float
    max_depth: Optional[float]
    num_levels: Optional[int]
    variables: Optional[List[str]]
    summary: Optional[str]


class FloatResponse(BaseModel):
    """Float response model."""

    float_id: int
    wmo_number: str
    platform_type: Optional[str]
    status: Optional[str]
    total_cycles: int
    last_update: Optional[str]


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "BluePy API",
        "version": "1.0.0",
        "description": "AI Conversational Interface for ARGO Data",
        "endpoints": {
            "chat": "/chat",
            "profiles": "/profiles",
            "profile_detail": "/profile/{id}",
            "floats": "/floats",
            "statistics": "/statistics",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint."""
    try:
        db.execute(text("SELECT 1"))
        
        orchestrator = get_orchestrator()
        
        return {
            "status": "healthy",
            "database": "connected",
            "rag_orchestrator": "initialized",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process natural language query through RAG pipeline.
    
    This endpoint:
    1. Retrieves relevant context from vector DB
    2. Translates query to SQL using MCP
    3. Executes query
    4. Returns results with visualization metadata
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        orchestrator = get_orchestrator()
        
        response = orchestrator.process_query(
            user_query=request.message,
            session_id=session_id,
            retrieve_context=request.retrieve_context,
            n_context=request.n_context,
        )
        
        return ChatResponse(**response)
    
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/profiles", response_model=List[ProfileResponse])
async def get_profiles(
    float_wmo: Optional[str] = Query(None, description="Filter by float WMO number"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    min_lat: Optional[float] = Query(None, description="Minimum latitude"),
    max_lat: Optional[float] = Query(None, description="Maximum latitude"),
    min_lon: Optional[float] = Query(None, description="Minimum longitude"),
    max_lon: Optional[float] = Query(None, description="Maximum longitude"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    db: Session = Depends(get_db),
):
    """Get profiles with optional filters."""
    query = db.query(ArgoProfileMeta)
    
    if float_wmo:
        query = query.filter(ArgoProfileMeta.float_wmo == float_wmo)
    if start_date:
        query = query.filter(ArgoProfileMeta.timestamp >= start_date)
    if end_date:
        query = query.filter(ArgoProfileMeta.timestamp <= end_date)
    
    if all([min_lat, max_lat, min_lon, max_lon]):
        query = query.filter(
            text(
                f"ST_Within(geom, ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, 4326))"
            )
        )
    
    profiles = query.limit(limit).all()
    
    return [
        ProfileResponse(
            profile_id=p.profile_id,
            float_wmo=p.float_wmo,
            cycle_number=p.cycle_number,
            timestamp=str(p.timestamp),
            latitude=p.latitude,
            longitude=p.longitude,
            max_depth=p.max_depth,
            num_levels=p.num_levels,
            variables=p.variables,
            summary=p.summary,
        )
        for p in profiles
    ]


@app.get("/profile/{profile_id}")
async def get_profile_detail(profile_id: int, db: Session = Depends(get_db)):
    """Get detailed profile data including measurements."""
    profile = db.query(ArgoProfileMeta).filter(ArgoProfileMeta.profile_id == profile_id).first()
    
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    data_query = text(
        """
        SELECT depths, pressures, temperatures, salinities, 
               dissolved_oxygens, chlorophylls
        FROM argo_profile_jsonb
        WHERE profile_id = :profile_id
        """
    )
    result = db.execute(data_query, {"profile_id": profile_id})
    data_row = result.fetchone()
    
    response = {
        "profile_id": profile.profile_id,
        "float_wmo": profile.float_wmo,
        "cycle_number": profile.cycle_number,
        "timestamp": str(profile.timestamp),
        "latitude": profile.latitude,
        "longitude": profile.longitude,
        "max_depth": profile.max_depth,
        "num_levels": profile.num_levels,
        "variables": profile.variables,
        "summary": profile.summary,
    }
    
    if data_row:
        response["data"] = {
            "depths": list(data_row[0]) if data_row[0] else [],
            "pressures": list(data_row[1]) if data_row[1] else [],
            "temperatures": list(data_row[2]) if data_row[2] else [],
            "salinities": list(data_row[3]) if data_row[3] else [],
            "dissolved_oxygens": list(data_row[4]) if data_row[4] else [],
            "chlorophylls": list(data_row[5]) if data_row[5] else [],
        }
    
    return response


@app.get("/floats", response_model=List[FloatResponse])
async def get_floats(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """Get list of floats."""
    query = db.query(ArgoFloat)
    
    if status:
        query = query.filter(ArgoFloat.status == status)
    
    floats = query.limit(limit).all()
    
    return [
        FloatResponse(
            float_id=f.float_id,
            wmo_number=f.wmo_number,
            platform_type=f.platform_type,
            status=f.status,
            total_cycles=f.total_cycles,
            last_update=str(f.last_update) if f.last_update else None,
        )
        for f in floats
    ]


@app.get("/statistics")
async def get_statistics():
    """Get database statistics."""
    try:
        orchestrator = get_orchestrator()
        stats = orchestrator.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/map/geojson")
async def get_map_geojson(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    float_wmo: Optional[str] = Query(None),
    limit: int = Query(1000, ge=1, le=10000),
    db: Session = Depends(get_db),
):
    """Get profile locations as GeoJSON for mapping."""
    query = db.query(ArgoProfileMeta)
    
    if float_wmo:
        query = query.filter(ArgoProfileMeta.float_wmo == float_wmo)
    if start_date:
        query = query.filter(ArgoProfileMeta.timestamp >= start_date)
    if end_date:
        query = query.filter(ArgoProfileMeta.timestamp <= end_date)
    
    profiles = query.limit(limit).all()
    
    features = []
    for p in profiles:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [p.longitude, p.latitude],
            },
            "properties": {
                "profile_id": p.profile_id,
                "float_wmo": p.float_wmo,
                "cycle_number": p.cycle_number,
                "timestamp": str(p.timestamp),
                "max_depth": p.max_depth,
                "variables": p.variables,
            },
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }
    
    return geojson


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    try:
        orchestrator = get_orchestrator()
        orchestrator.clear_session(session_id)
        return {"success": True, "message": f"Session {session_id} cleared"}
    except Exception as e:
        logger.error(f"Clear session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("API_RELOAD", "true").lower() == "true",
    )
