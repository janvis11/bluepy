"""Main data ingestion pipeline for ARGO data."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from geoalchemy2.shape import from_shape
from loguru import logger
from shapely.geometry import Point
from sqlalchemy.orm import Session

from backend.db.database import SessionLocal, init_db
from backend.db.models import ArgoFloat, ArgoProfileData, ArgoProfileJsonb, ArgoProfileMeta
from ingestion.parsers.netcdf_parser import ArgoProfile, NetCDFParser


class ArgoIngestionPipeline:
    """Pipeline for ingesting ARGO NetCDF data into database and Parquet."""

    def __init__(self, output_dir: str = "./data/processed", batch_size: int = 100, max_workers: int = 4):
        """
        Initialize ingestion pipeline.

        Args:
            output_dir: Directory for processed data (Parquet files)
            batch_size: Number of profiles to batch before writing
            max_workers: Number of parallel workers for parsing
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir = self.output_dir / "parquet"
        self.parquet_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.parser = NetCDFParser()
        logger.info(f"Initialized pipeline: output={output_dir}, batch_size={batch_size}, workers={max_workers}")

    def ingest_directory(self, input_dir: str, recursive: bool = True):
        """
        Ingest all NetCDF files from a directory.

        Args:
            input_dir: Directory containing NetCDF files
            recursive: Whether to search recursively
        """
        logger.info(f"Starting ingestion from directory: {input_dir}")

        netcdf_files = self._find_netcdf_files(input_dir, recursive)
        logger.info(f"Found {len(netcdf_files)} NetCDF files")

        if not netcdf_files:
            logger.warning("No NetCDF files found!")
            return

        all_profiles = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self.parser.parse_file, str(f)): f for f in netcdf_files}

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    profiles = future.result()
                    all_profiles.extend(profiles)
                    logger.info(f"Parsed {len(profiles)} profiles from {file_path.name}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

        logger.info(f"Total profiles parsed: {len(all_profiles)}")

        if all_profiles:
            self._store_profiles(all_profiles)

    def ingest_file(self, file_path: str):
        """
        Ingest a single NetCDF file.

        Args:
            file_path: Path to NetCDF file
        """
        logger.info(f"Ingesting single file: {file_path}")
        profiles = self.parser.parse_file(file_path)
        if profiles:
            self._store_profiles(profiles)

    def _find_netcdf_files(self, directory: str, recursive: bool) -> List[Path]:
        """Find all NetCDF files in directory."""
        path = Path(directory)
        if recursive:
            return list(path.rglob("*.nc"))
        else:
            return list(path.glob("*.nc"))

    def _store_profiles(self, profiles: List[ArgoProfile]):
        """Store profiles in database and Parquet."""
        logger.info(f"Storing {len(profiles)} profiles...")

        self._store_in_database(profiles)

        self._store_in_parquet(profiles)

        logger.info("Storage complete")

    def _store_in_database(self, profiles: List[ArgoProfile]):
        """Store profiles in PostgreSQL database."""
        db = SessionLocal()
        try:
            floats_map = {}
            for profile in profiles:
                if profile.float_wmo not in floats_map:
                    floats_map[profile.float_wmo] = []
                floats_map[profile.float_wmo].append(profile)

            for wmo, float_profiles in floats_map.items():
                self._upsert_float(db, wmo, float_profiles)

            for i in range(0, len(profiles), self.batch_size):
                batch = profiles[i : i + self.batch_size]
                self._insert_profile_batch(db, batch)
                db.commit()
                logger.info(f"Committed batch {i

        except Exception as e:
            db.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            db.close()

    def _upsert_float(self, db: Session, wmo: str, profiles: List[ArgoProfile]):
        """Insert or update float metadata."""
        float_obj = db.query(ArgoFloat).filter(ArgoFloat.wmo_number == wmo).first()

        if not float_obj:
            float_obj = ArgoFloat(
                wmo_number=wmo,
                platform_type=profiles[0].platform_type if profiles else None,
                deployment_date=min(p.timestamp for p in profiles),
                last_update=max(p.timestamp for p in profiles),
                status="active",
                total_cycles=len(profiles),
            )
            db.add(float_obj)
            logger.debug(f"Created new float: {wmo}")
        else:
            float_obj.last_update = max(p.timestamp for p in profiles)
            float_obj.total_cycles += len(profiles)
            logger.debug(f"Updated float: {wmo}")

    def _insert_profile_batch(self, db: Session, profiles: List[ArgoProfile]):
        """Insert a batch of profiles."""
        for profile in profiles:
            existing = (
                db.query(ArgoProfileMeta)
                .filter(
                    ArgoProfileMeta.float_wmo == profile.float_wmo,
                    ArgoProfileMeta.cycle_number == profile.cycle_number,
                )
                .first()
            )

            if existing:
                logger.debug(f"Profile already exists: {profile.float_wmo} cycle {profile.cycle_number}")
                continue

            point = Point(profile.longitude, profile.latitude)
            profile_meta = ArgoProfileMeta(
                float_wmo=profile.float_wmo,
                cycle_number=profile.cycle_number,
                timestamp=profile.timestamp,
                geom=from_shape(point, srid=4326),
                latitude=profile.latitude,
                longitude=profile.longitude,
                max_depth=profile.max_depth,
                num_levels=profile.num_levels,
                variables=profile.available_variables,
                qc_status="pending",
                source_file=profile.source_file,
                summary=profile.generate_summary(),
            )
            db.add(profile_meta)
            db.flush() 

            profile_jsonb = ArgoProfileJsonb(
                profile_id=profile_meta.profile_id,
                depths=profile.depths,
                pressures=profile.pressures,
                temperatures=profile.temperatures,
                temperatures_qc=profile.temperatures_qc,
                salinities=profile.salinities,
                salinities_qc=profile.salinities_qc,
                dissolved_oxygens=profile.dissolved_oxygens,
                dissolved_oxygens_qc=profile.dissolved_oxygens_qc,
                chlorophylls=profile.chlorophylls,
                chlorophylls_qc=profile.chlorophylls_qc,
            )
            db.add(profile_jsonb)


    def _insert_profile_data_normalized(self, db: Session, profile_id: int, profile: ArgoProfile):
        """Insert profile data in normalized format (one row per depth)."""
        for i, depth in enumerate(profile.depths):
            data_point = ArgoProfileData(
                profile_id=profile_id,
                depth=depth,
                pressure=profile.pressures[i] if profile.pressures else None,
                temperature=profile.temperatures[i] if profile.temperatures and i < len(profile.temperatures) else None,
                temperature_qc=(
                    profile.temperatures_qc[i] if profile.temperatures_qc and i < len(profile.temperatures_qc) else None
                ),
                salinity=profile.salinities[i] if profile.salinities and i < len(profile.salinities) else None,
                salinity_qc=profile.salinities_qc[i] if profile.salinities_qc and i < len(profile.salinities_qc) else None,
                dissolved_oxygen=(
                    profile.dissolved_oxygens[i] if profile.dissolved_oxygens and i < len(profile.dissolved_oxygens) else None
                ),
                dissolved_oxygen_qc=(
                    profile.dissolved_oxygens_qc[i]
                    if profile.dissolved_oxygens_qc and i < len(profile.dissolved_oxygens_qc)
                    else None
                ),
                chlorophyll=profile.chlorophylls[i] if profile.chlorophylls and i < len(profile.chlorophylls) else None,
                chlorophyll_qc=(
                    profile.chlorophylls_qc[i] if profile.chlorophylls_qc and i < len(profile.chlorophylls_qc) else None
                ),
            )
            db.add(data_point)

    def _store_in_parquet(self, profiles: List[ArgoProfile]):
        """Store profiles in Parquet format for analytics."""
        records = []
        for profile in profiles:
            for i, depth in enumerate(profile.depths):
                record = {
                    "float_wmo": profile.float_wmo,
                    "cycle_number": profile.cycle_number,
                    "timestamp": profile.timestamp,
                    "latitude": profile.latitude,
                    "longitude": profile.longitude,
                    "depth": depth,
                    "pressure": profile.pressures[i] if profile.pressures else None,
                    "temperature": profile.temperatures[i] if profile.temperatures and i < len(profile.temperatures) else None,
                    "temperature_qc": (
                        profile.temperatures_qc[i] if profile.temperatures_qc and i < len(profile.temperatures_qc) else None
                    ),
                    "salinity": profile.salinities[i] if profile.salinities and i < len(profile.salinities) else None,
                    "salinity_qc": (
                        profile.salinities_qc[i] if profile.salinities_qc and i < len(profile.salinities_qc) else None
                    ),
                    "dissolved_oxygen": (
                        profile.dissolved_oxygens[i]
                        if profile.dissolved_oxygens and i < len(profile.dissolved_oxygens)
                        else None
                    ),
                    "chlorophyll": profile.chlorophylls[i] if profile.chlorophylls and i < len(profile.chlorophylls) else None,
                }
                records.append(record)

        if not records:
            logger.warning("No records to write to Parquet")
            return

        df = pd.DataFrame(records)

        output_file = self.parquet_dir / f"argo_profiles_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file, compression="snappy")
        logger.info(f"Wrote Parquet file: {output_file}")


def main():
    """Main entry point for ingestion pipeline."""
    parser = argparse.ArgumentParser(description="ARGO data ingestion pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input directory or file")
    parser.add_argument("--output", "-o", default="./data/processed", help="Output directory")
    parser.add_argument("--batch-size", "-b", type=int, default=100, help="Batch size for database inserts")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--init-db", action="store_true", help="Initialize database before ingestion")
    parser.add_argument("--recursive", "-r", action="store_true", help="Search recursively for NetCDF files")

    args = parser.parse_args()

    if args.init_db:
        logger.info("Initializing database...")
        init_db()

    pipeline = ArgoIngestionPipeline(output_dir=args.output, batch_size=args.batch_size, max_workers=args.workers)

    input_path = Path(args.input)
    if input_path.is_file():
        pipeline.ingest_file(str(input_path))
    elif input_path.is_dir():
        pipeline.ingest_directory(str(input_path), recursive=args.recursive)
    else:
        logger.error(f"Invalid input path: {args.input}")
        return 1

    logger.info("Ingestion complete!")
    return 0


if __name__ == "__main__":
    exit(main())
