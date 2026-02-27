"""Generate sample ARGO data for testing and demonstration."""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from random import Random

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoalchemy2.shape import from_shape
from loguru import logger
from shapely.geometry import Point

from backend.db.database import get_db_context
from backend.db.models import ArgoFloat, ArgoProfileJsonb, ArgoProfileMeta


class SampleDataGenerator:
    """Generate sample ARGO profile data."""

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        self.rng = Random(seed)
        np.random.seed(seed)

    def generate_profile(
        self,
        float_wmo: str,
        cycle: int,
        base_lat: float,
        base_lon: float,
        timestamp: datetime,
    ) -> dict:
        """Generate a single profile."""
        lat = base_lat + self.rng.uniform(-2, 2)
        lon = base_lon + self.rng.uniform(-2, 2)

        depths = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000]
        num_levels = len(depths)

        surface_temp = 25 + self.rng.uniform(-3, 3)
        temperatures = []
        for depth in depths:
            if depth < 100:
                temp = surface_temp - (depth / 100) * 3
            elif depth < 500:
                temp = surface_temp - 3 - ((depth - 100) / 400) * 10
            else:
                temp = surface_temp - 13 - ((depth - 500) / 1500) * 7
            temp += self.rng.uniform(-0.5, 0.5)
            temperatures.append(round(temp, 2))

        base_salinity = 34.5 + self.rng.uniform(-0.5, 0.5)
        salinities = []
        for depth in depths:
            sal = base_salinity + (depth / 2000) * 0.5 + self.rng.uniform(-0.1, 0.1)
            salinities.append(round(sal, 2))

        surface_do = 220 + self.rng.uniform(-20, 20)
        dissolved_oxygens = []
        for depth in depths:
            if depth < 100:
                do = surface_do - (depth / 100) * 50
            elif depth < 800:
                do = surface_do - 50 - ((depth - 100) / 700) * 100
            else:
                do = surface_do - 150 + ((depth - 800) / 1200) * 30
            do += self.rng.uniform(-10, 10)
            dissolved_oxygens.append(round(max(do, 10), 2))

        chlorophylls = []
        for depth in depths:
            if depth < 50:
                chl = 0.5 + self.rng.uniform(-0.2, 0.2)
            elif depth < 150:
                chl = 0.5 - ((depth - 50) / 100) * 0.4
            else:
                chl = 0.1 + self.rng.uniform(-0.05, 0.05)
            chlorophylls.append(round(max(chl, 0.01), 3))

        qc_flags = [1 if self.rng.random() > 0.05 else 3 for _ in depths]

        summary = (
            f"Float {float_wmo}, Cycle {cycle}, "
            f"Date: {timestamp.strftime('%Y-%m-%d')}, "
            f"Location: ({lat:.2f}°N, {lon:.2f}°E), "
            f"Max depth: {max(depths):.0f}m. "
            f"Surface temp: {temperatures[0]:.1f}°C, "
            f"Surface salinity: {salinities[0]:.1f} psu, "
            f"Surface DO: {dissolved_oxygens[0]:.1f} µmol/kg."
        )

        return {
            "float_wmo": float_wmo,
            "cycle_number": cycle,
            "timestamp": timestamp,
            "latitude": lat,
            "longitude": lon,
            "depths": depths,
            "temperatures": temperatures,
            "salinities": salinities,
            "dissolved_oxygens": dissolved_oxygens,
            "chlorophylls": chlorophylls,
            "qc_flags": qc_flags,
            "max_depth": max(depths),
            "num_levels": num_levels,
            "summary": summary,
        }

    def generate_float_trajectory(
        self,
        float_wmo: str,
        num_cycles: int,
        start_lat: float,
        start_lon: float,
        start_date: datetime,
    ) -> list:
        """Generate a trajectory of profiles for a float."""
        profiles = []
        current_lat = start_lat
        current_lon = start_lon

        for cycle in range(1, num_cycles + 1):
            current_lat += self.rng.uniform(-0.5, 0.5)
            current_lon += self.rng.uniform(-0.5, 0.5)

            timestamp = start_date + timedelta(days=10 * cycle)

            profile = self.generate_profile(float_wmo, cycle, current_lat, current_lon, timestamp)
            profiles.append(profile)

        return profiles


def main():
    """Generate and insert sample data."""
    parser = argparse.ArgumentParser(description="Generate sample ARGO data")
    parser.add_argument("--num-floats", type=int, default=15, help="Number of floats to generate")
    parser.add_argument("--cycles-per-float", type=int, default=25, help="Number of cycles per float")
    parser.add_argument("--region", default="all", choices=["indian_ocean", "pacific", "atlantic", "all"])
    args = parser.parse_args()

    regions = {
        "indian_ocean": {"lat_range": (-10, 20), "lon_range": (40, 100)},
        "pacific": {"lat_range": (-20, 20), "lon_range": (120, 180)},
        "atlantic": {"lat_range": (-10, 30), "lon_range": (-50, 10)},
    }

    if args.region == "all":
        region_list = list(regions.keys())
        logger.info(f"Generating {args.num_floats} floats across all regions with {args.cycles_per_float} cycles each")
    else:
        region_list = [args.region]
        logger.info(f"Generating {args.num_floats} floats in {args.region} with {args.cycles_per_float} cycles each")

    generator = SampleDataGenerator()
    total_profiles = 0

    with get_db_context() as db:
        for float_idx in range(args.num_floats):
            float_wmo = f"290{2000 + float_idx}"

            if args.region == "all":
                region_name = region_list[float_idx % len(region_list)]
                region_config = regions[region_name]
            else:
                region_name = args.region
                region_config = regions[region_name]

            start_lat = generator.rng.uniform(*region_config["lat_range"])
            start_lon = generator.rng.uniform(*region_config["lon_range"])
            start_date = datetime.now() - timedelta(days=365)

            float_obj = ArgoFloat(
                wmo_number=float_wmo,
                platform_type="APEX",
                deployment_date=start_date,
                status="active",
            )
            db.add(float_obj)

            profiles = generator.generate_float_trajectory(
                float_wmo, args.cycles_per_float, start_lat, start_lon, start_date
            )

            for profile in profiles:
                point = Point(profile["longitude"], profile["latitude"])
                profile_meta = ArgoProfileMeta(
                    float_wmo=profile["float_wmo"],
                    cycle_number=profile["cycle_number"],
                    timestamp=profile["timestamp"],
                    geom=from_shape(point, srid=4326),
                    latitude=profile["latitude"],
                    longitude=profile["longitude"],
                    max_depth=profile["max_depth"],
                    num_levels=profile["num_levels"],
                    variables=["temperature", "salinity", "dissolved_oxygen", "chlorophyll"],
                    qc_status="good",
                    summary=profile["summary"],
                )
                db.add(profile_meta)
                db.flush()

                profile_jsonb = ArgoProfileJsonb(
                    profile_id=profile_meta.profile_id,
                    depths=profile["depths"],
                    temperatures=profile["temperatures"],
                    temperatures_qc=profile["qc_flags"],
                    salinities=profile["salinities"],
                    salinities_qc=profile["qc_flags"],
                    dissolved_oxygens=profile["dissolved_oxygens"],
                    dissolved_oxygens_qc=profile["qc_flags"],
                    chlorophylls=profile["chlorophylls"],
                    chlorophylls_qc=profile["qc_flags"],
                )
                db.add(profile_jsonb)

            total_profiles += len(profiles)
            logger.info(f"Generated float {float_wmo} in {region_name} with {len(profiles)} profiles")

        db.commit()

    logger.success(f"✓ Sample data generation complete! Generated {total_profiles} profiles from {args.num_floats} floats")
    return 0


if __name__ == "__main__":
    exit(main())
