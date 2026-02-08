CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS argo_float (
    float_id SERIAL PRIMARY KEY,
    wmo_number VARCHAR(20) UNIQUE NOT NULL,
    platform_type VARCHAR(50),
    deployment_date TIMESTAMP,
    last_update TIMESTAMP,
    status VARCHAR(20),  
    total_cycles INTEGER DEFAULT 0,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_argo_float_wmo ON argo_float(wmo_number);
CREATE INDEX idx_argo_float_status ON argo_float(status);

CREATE TABLE IF NOT EXISTS argo_profile_meta (
    profile_id SERIAL PRIMARY KEY,
    float_wmo VARCHAR(20) NOT NULL,
    cycle_number INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    geom GEOMETRY(POINT, 4326) NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    max_depth FLOAT,
    num_levels INTEGER,
    variables TEXT[],  
    qc_status VARCHAR(20),
    source_file VARCHAR(500),
    summary TEXT,  
    embedding_id VARCHAR(100),  
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(float_wmo, cycle_number)
);

CREATE INDEX idx_profile_meta_geom ON argo_profile_meta USING GIST (geom);
CREATE INDEX idx_profile_meta_timestamp ON argo_profile_meta(timestamp);
CREATE INDEX idx_profile_meta_float ON argo_profile_meta(float_wmo);
CREATE INDEX idx_profile_meta_cycle ON argo_profile_meta(cycle_number);
CREATE INDEX idx_profile_meta_variables ON argo_profile_meta USING GIN (variables);

CREATE TABLE IF NOT EXISTS argo_profile_data (
    data_id BIGSERIAL PRIMARY KEY,
    profile_id INTEGER NOT NULL REFERENCES argo_profile_meta(profile_id) ON DELETE CASCADE,
    depth FLOAT NOT NULL,
    pressure FLOAT,
    temperature FLOAT,
    temperature_qc INTEGER,
    salinity FLOAT,
    salinity_qc INTEGER,
    dissolved_oxygen FLOAT,
    dissolved_oxygen_qc INTEGER,
    chlorophyll FLOAT,
    chlorophyll_qc INTEGER,
    nitrate FLOAT,
    nitrate_qc INTEGER,
    ph FLOAT,
    ph_qc INTEGER,
    additional_data JSONB,  
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_profile_data_profile ON argo_profile_data(profile_id);
CREATE INDEX idx_profile_data_depth ON argo_profile_data(depth);
CREATE INDEX idx_profile_data_temp ON argo_profile_data(temperature) WHERE temperature IS NOT NULL;
CREATE INDEX idx_profile_data_sal ON argo_profile_data(salinity) WHERE salinity IS NOT NULL;

CREATE TABLE IF NOT EXISTS argo_profile_jsonb (
    profile_id INTEGER PRIMARY KEY REFERENCES argo_profile_meta(profile_id) ON DELETE CASCADE,
    depths FLOAT[],
    pressures FLOAT[],
    temperatures FLOAT[],
    temperatures_qc INTEGER[],
    salinities FLOAT[],
    salinities_qc INTEGER[],
    dissolved_oxygens FLOAT[],
    dissolved_oxygens_qc INTEGER[],
    chlorophylls FLOAT[],
    chlorophylls_qc INTEGER[],
    other_variables JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS query_history (
    query_id SERIAL PRIMARY KEY,
    session_id VARCHAR(100),
    user_query TEXT NOT NULL,
    query_embedding_id VARCHAR(100),
    generated_sql TEXT,
    sql_validated BOOLEAN,
    execution_time_ms INTEGER,
    result_count INTEGER,
    visualization_type VARCHAR(50),
    success BOOLEAN,
    error_message TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_query_history_session ON query_history(session_id);
CREATE INDEX idx_query_history_timestamp ON query_history(timestamp);

CREATE TABLE IF NOT EXISTS embeddings_metadata (
    embedding_id VARCHAR(100) PRIMARY KEY,
    content_type VARCHAR(50) NOT NULL,  
    content_text TEXT NOT NULL,
    reference_id INTEGER,  
    vector_db_id VARCHAR(200),  
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_embeddings_type ON embeddings_metadata(content_type);
CREATE INDEX idx_embeddings_ref ON embeddings_metadata(reference_id);

CREATE OR REPLACE FUNCTION get_profiles_in_bbox(
    min_lon DOUBLE PRECISION,
    min_lat DOUBLE PRECISION,
    max_lon DOUBLE PRECISION,
    max_lat DOUBLE PRECISION,
    start_date TIMESTAMP DEFAULT NULL,
    end_date TIMESTAMP DEFAULT NULL
)
RETURNS TABLE (
    profile_id INTEGER,
    float_wmo VARCHAR,
    cycle_number INTEGER,
    timestamp TIMESTAMP,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pm.profile_id,
        pm.float_wmo,
        pm.cycle_number,
        pm.timestamp,
        pm.latitude,
        pm.longitude
    FROM argo_profile_meta pm
    WHERE ST_Within(
        pm.geom,
        ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326)
    )
    AND (start_date IS NULL OR pm.timestamp >= start_date)
    AND (end_date IS NULL OR pm.timestamp <= end_date);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_profiles_near_point(
    center_lon DOUBLE PRECISION,
    center_lat DOUBLE PRECISION,
    radius_meters DOUBLE PRECISION,
    start_date TIMESTAMP DEFAULT NULL,
    end_date TIMESTAMP DEFAULT NULL
)
RETURNS TABLE (
    profile_id INTEGER,
    float_wmo VARCHAR,
    cycle_number INTEGER,
    timestamp TIMESTAMP,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    distance_meters DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pm.profile_id,
        pm.float_wmo,
        pm.cycle_number,
        pm.timestamp,
        pm.latitude,
        pm.longitude,
        ST_Distance(
            pm.geom::geography,
            ST_SetSRID(ST_Point(center_lon, center_lat), 4326)::geography
        ) as distance_meters
    FROM argo_profile_meta pm
    WHERE ST_DWithin(
        pm.geom::geography,
        ST_SetSRID(ST_Point(center_lon, center_lat), 4326)::geography,
        radius_meters
    )
    AND (start_date IS NULL OR pm.timestamp >= start_date)
    AND (end_date IS NULL OR pm.timestamp <= end_date)
    ORDER BY distance_meters;
END;
$$ LANGUAGE plpgsql;

CREATE MATERIALIZED VIEW IF NOT EXISTS argo_statistics AS
SELECT 
    COUNT(DISTINCT float_wmo) as total_floats,
    COUNT(*) as total_profiles,
    MIN(timestamp) as earliest_date,
    MAX(timestamp) as latest_date,
    AVG(max_depth) as avg_max_depth,
    COUNT(*) FILTER (WHERE 'temperature' = ANY(variables)) as profiles_with_temp,
    COUNT(*) FILTER (WHERE 'salinity' = ANY(variables)) as profiles_with_sal,
    COUNT(*) FILTER (WHERE 'dissolved_oxygen' = ANY(variables)) as profiles_with_do
FROM argo_profile_meta;

CREATE UNIQUE INDEX ON argo_statistics ((true));

CREATE OR REPLACE FUNCTION refresh_argo_statistics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY argo_statistics;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION update_float_last_update()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE argo_float
    SET last_update = NEW.timestamp,
        total_cycles = total_cycles + 1,
        updated_at = CURRENT_TIMESTAMP
    WHERE wmo_number = NEW.float_wmo;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_float_last_update
AFTER INSERT ON argo_profile_meta
FOR EACH ROW
EXECUTE FUNCTION update_float_last_update();

COMMENT ON TABLE argo_float IS 'Metadata for ARGO floats';
COMMENT ON TABLE argo_profile_meta IS 'Metadata for each profile/cycle with spatial indexing';
COMMENT ON TABLE argo_profile_data IS 'Normalized profile measurements (one row per depth level)';
COMMENT ON TABLE argo_profile_jsonb IS 'Profile measurements stored as arrays in JSONB for fast retrieval';
COMMENT ON TABLE query_history IS 'History of user queries for analytics and caching';
COMMENT ON TABLE embeddings_metadata IS 'Metadata linking to vector database embeddings';
