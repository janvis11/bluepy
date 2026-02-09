"""SQLAlchemy ORM models for ARGO database."""

from datetime import datetime
from typing import List, Optional

from geoalchemy2 import Geometry
from sqlalchemy import (
    ARRAY,
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class ArgoFloat(Base):
    """ARGO float metadata."""

    __tablename__ = "argo_float"

    float_id = Column(Integer, primary_key=True, autoincrement=True)
    wmo_number = Column(String(20), unique=True, nullable=False, index=True)
    platform_type = Column(String(50))
    deployment_date = Column(DateTime)
    last_update = Column(DateTime)
    status = Column(String(20), index=True) 
    total_cycles = Column(Integer, default=0)
    metadata = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    profiles = relationship("ArgoProfileMeta", back_populates="float", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ArgoFloat(wmo={self.wmo_number}, status={self.status})>"


class ArgoProfileMeta(Base):
    """Profile metadata with spatial information."""

    __tablename__ = "argo_profile_meta"

    profile_id = Column(Integer, primary_key=True, autoincrement=True)
    float_wmo = Column(String(20), ForeignKey("argo_float.wmo_number"), nullable=False, index=True)
    cycle_number = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    geom = Column(Geometry("POINT", srid=4326), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    max_depth = Column(Float)
    num_levels = Column(Integer)
    variables = Column(ARRAY(Text)) 
    qc_status = Column(String(20))
    source_file = Column(String(500))
    summary = Column(Text) 
    embedding_id = Column(String(100))
    metadata = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("float_wmo", "cycle_number", name="uq_float_cycle"),)

    float = relationship("ArgoFloat", back_populates="profiles")
    data_points = relationship("ArgoProfileData", back_populates="profile", cascade="all, delete-orphan")
    jsonb_data = relationship(
        "ArgoProfileJsonb", back_populates="profile", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<ArgoProfileMeta(id={self.profile_id}, float={self.float_wmo}, cycle={self.cycle_number})>"


class ArgoProfileData(Base):
    """Profile measurements at each depth level (normalized)."""

    __tablename__ = "argo_profile_data"

    data_id = Column(BigInteger, primary_key=True, autoincrement=True)
    profile_id = Column(Integer, ForeignKey("argo_profile_meta.profile_id", ondelete="CASCADE"), nullable=False, index=True)
    depth = Column(Float, nullable=False, index=True)
    pressure = Column(Float)
    temperature = Column(Float, index=True)
    temperature_qc = Column(Integer)
    salinity = Column(Float, index=True)
    salinity_qc = Column(Integer)
    dissolved_oxygen = Column(Float)
    dissolved_oxygen_qc = Column(Integer)
    chlorophyll = Column(Float)
    chlorophyll_qc = Column(Integer)
    nitrate = Column(Float)
    nitrate_qc = Column(Integer)
    ph = Column(Float)
    ph_qc = Column(Integer)
    additional_data = Column(JSONB) 
    created_at = Column(DateTime, default=datetime.utcnow)

    profile = relationship("ArgoProfileMeta", back_populates="data_points")

    def __repr__(self):
        return f"<ArgoProfileData(profile={self.profile_id}, depth={self.depth})>"


class ArgoProfileJsonb(Base):
    """Profile measurements stored as arrays (alternative storage for fast retrieval)."""

    __tablename__ = "argo_profile_jsonb"

    profile_id = Column(Integer, ForeignKey("argo_profile_meta.profile_id", ondelete="CASCADE"), primary_key=True)
    depths = Column(ARRAY(Float))
    pressures = Column(ARRAY(Float))
    temperatures = Column(ARRAY(Float))
    temperatures_qc = Column(ARRAY(Integer))
    salinities = Column(ARRAY(Float))
    salinities_qc = Column(ARRAY(Integer))
    dissolved_oxygens = Column(ARRAY(Float))
    dissolved_oxygens_qc = Column(ARRAY(Integer))
    chlorophylls = Column(ARRAY(Float))
    chlorophylls_qc = Column(ARRAY(Integer))
    other_variables = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)

    profile = relationship("ArgoProfileMeta", back_populates="jsonb_data")

    def __repr__(self):
        return f"<ArgoProfileJsonb(profile={self.profile_id}, levels={len(self.depths) if self.depths else 0})>"


class QueryHistory(Base):
    """History of user queries for analytics and caching."""

    __tablename__ = "query_history"

    query_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), index=True)
    user_query = Column(Text, nullable=False)
    query_embedding_id = Column(String(100))
    generated_sql = Column(Text)
    sql_validated = Column(Boolean)
    execution_time_ms = Column(Integer)
    result_count = Column(Integer)
    visualization_type = Column(String(50))
    success = Column(Boolean)
    error_message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<QueryHistory(id={self.query_id}, success={self.success})>"


class EmbeddingsMetadata(Base):
    """Metadata linking to vector database embeddings."""

    __tablename__ = "embeddings_metadata"

    embedding_id = Column(String(100), primary_key=True)
    content_type = Column(String(50), nullable=False, index=True) 
    content_text = Column(Text, nullable=False)
    reference_id = Column(Integer, index=True) 
    vector_db_id = Column(String(200)) 
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<EmbeddingsMetadata(id={self.embedding_id}, type={self.content_type})>"
