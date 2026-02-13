"""NetCDF parser for ARGO float data."""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger


@dataclass
class ArgoProfile:
    """Parsed ARGO profile data."""

    float_wmo: str
    cycle_number: int
    timestamp: datetime
    latitude: float
    longitude: float
    depths: List[float]
    pressures: List[float]
    temperatures: Optional[List[float]] = None
    temperatures_qc: Optional[List[int]] = None
    salinities: Optional[List[float]] = None
    salinities_qc: Optional[List[int]] = None
    dissolved_oxygens: Optional[List[float]] = None
    dissolved_oxygens_qc: Optional[List[int]] = None
    chlorophylls: Optional[List[float]] = None
    chlorophylls_qc: Optional[List[int]] = None
    nitrates: Optional[List[float]] = None
    nitrates_qc: Optional[List[int]] = None
    ph_values: Optional[List[float]] = None
    ph_qc: Optional[List[int]] = None
    platform_type: Optional[str] = None
    source_file: Optional[str] = None
    metadata: Optional[Dict] = None

    @property
    def max_depth(self) -> float:
        """Get maximum depth in profile."""
        return max(self.depths) if self.depths else 0.0

    @property
    def num_levels(self) -> int:
        """Get number of depth levels."""
        return len(self.depths)

    @property
    def available_variables(self) -> List[str]:
        """Get list of available variables."""
        variables = []
        if self.temperatures is not None:
            variables.append("temperature")
        if self.salinities is not None:
            variables.append("salinity")
        if self.dissolved_oxygens is not None:
            variables.append("dissolved_oxygen")
        if self.chlorophylls is not None:
            variables.append("chlorophyll")
        if self.nitrates is not None:
            variables.append("nitrate")
        if self.ph_values is not None:
            variables.append("ph")
        return variables

    def generate_summary(self) -> str:
        """Generate human-readable summary for RAG."""
        summary_parts = [
            f"Float {self.float_wmo}",
            f"Cycle {self.cycle_number}",
            f"Date: {self.timestamp.strftime('%Y-%m-%d %H:%M')}",
            f"Location: ({self.latitude:.2f}°N, {self.longitude:.2f}°E)",
            f"Max depth: {self.max_depth:.1f}m",
        ]

        if self.temperatures:
            valid_temps = [t for t in self.temperatures if t is not None and not np.isnan(t)]
            if valid_temps:
                summary_parts.append(f"Surface temp: {valid_temps[0]:.2f}°C")

        if self.salinities:
            valid_sals = [s for s in self.salinities if s is not None and not np.isnan(s)]
            if valid_sals:
                summary_parts.append(f"Surface salinity: {valid_sals[0]:.2f} psu")

        if self.dissolved_oxygens:
            valid_do = [d for d in self.dissolved_oxygens if d is not None and not np.isnan(d)]
            if valid_do:
                summary_parts.append(f"Surface DO: {valid_do[0]:.2f} µmol/kg")

        return ". ".join(summary_parts) + "."


class NetCDFParser:
    """Parser for ARGO NetCDF files."""

    VAR_MAPPINGS = {
        "temperature": ["TEMP", "TEMP_ADJUSTED", "temperature", "temp"],
        "salinity": ["PSAL", "PSAL_ADJUSTED", "salinity", "sal"],
        "pressure": ["PRES", "PRES_ADJUSTED", "pressure", "pres"],
        "dissolved_oxygen": ["DOXY", "DOXY_ADJUSTED", "dissolved_oxygen", "oxygen"],
        "chlorophyll": ["CHLA", "CHLA_ADJUSTED", "chlorophyll", "chla"],
        "nitrate": ["NITRATE", "NITRATE_ADJUSTED", "nitrate"],
        "ph": ["PH_IN_SITU_TOTAL", "PH_IN_SITU_TOTAL_ADJUSTED", "ph"],
    }

    def __init__(self):
        """Initialize parser."""
        self.logger = logger

    def parse_file(self, file_path: str) -> List[ArgoProfile]:
        """
        Parse a single NetCDF file and extract all profiles.

        Args:
            file_path: Path to NetCDF file

        Returns:
            List of ArgoProfile objects
        """
        try:
            self.logger.info(f"Parsing NetCDF file: {file_path}")
            ds = xr.open_dataset(file_path)
            profiles = self._extract_profiles(ds, file_path)
            ds.close()
            self.logger.info(f"Extracted {len(profiles)} profiles from {file_path}")
            return profiles
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {e}")
            return []

    def _extract_profiles(self, ds: xr.Dataset, source_file: str) -> List[ArgoProfile]:
        """Extract profiles from xarray Dataset."""
        profiles = []

        float_wmo = self._get_wmo_number(ds)

        n_profiles = ds.dims.get("N_PROF", 1)

        for prof_idx in range(n_profiles):
            try:
                profile = self._extract_single_profile(ds, prof_idx, float_wmo, source_file)
                if profile:
                    profiles.append(profile)
            except Exception as e:
                self.logger.warning(f"Error extracting profile {prof_idx} from {source_file}: {e}")

        return profiles

    def _extract_single_profile(
        self, ds: xr.Dataset, prof_idx: int, float_wmo: str, source_file: str
    ) -> Optional[ArgoProfile]:
        """Extract a single profile from dataset."""
        cycle_number = self._get_value(ds, "CYCLE_NUMBER", prof_idx, default=prof_idx)

        timestamp = self._get_timestamp(ds, prof_idx)
        if not timestamp:
            return None

        latitude = self._get_value(ds, "LATITUDE", prof_idx)
        longitude = self._get_value(ds, "LONGITUDE", prof_idx)
        if latitude is None or longitude is None:
            return None

        pressures = self._get_variable_array(ds, "pressure", prof_idx)
        if pressures is None or len(pressures) == 0:
            return None

        depths = [p / 10.0 if p is not None else None for p in pressures]

        temperatures = self._get_variable_array(ds, "temperature", prof_idx)
        temperatures_qc = self._get_qc_array(ds, "temperature", prof_idx)
        salinities = self._get_variable_array(ds, "salinity", prof_idx)
        salinities_qc = self._get_qc_array(ds, "salinity", prof_idx)
        dissolved_oxygens = self._get_variable_array(ds, "dissolved_oxygen", prof_idx)
        dissolved_oxygens_qc = self._get_qc_array(ds, "dissolved_oxygen", prof_idx)
        chlorophylls = self._get_variable_array(ds, "chlorophyll", prof_idx)
        chlorophylls_qc = self._get_qc_array(ds, "chlorophyll", prof_idx)
        nitrates = self._get_variable_array(ds, "nitrate", prof_idx)
        nitrates_qc = self._get_qc_array(ds, "nitrate", prof_idx)
        ph_values = self._get_variable_array(ds, "ph", prof_idx)
        ph_qc = self._get_qc_array(ds, "ph", prof_idx)

        platform_type = self._get_platform_type(ds)

        profile = ArgoProfile(
            float_wmo=float_wmo,
            cycle_number=int(cycle_number),
            timestamp=timestamp,
            latitude=float(latitude),
            longitude=float(longitude),
            depths=depths,
            pressures=pressures,
            temperatures=temperatures,
            temperatures_qc=temperatures_qc,
            salinities=salinities,
            salinities_qc=salinities_qc,
            dissolved_oxygens=dissolved_oxygens,
            dissolved_oxygens_qc=dissolved_oxygens_qc,
            chlorophylls=chlorophylls,
            chlorophylls_qc=chlorophylls_qc,
            nitrates=nitrates,
            nitrates_qc=nitrates_qc,
            ph_values=ph_values,
            ph_qc=ph_qc,
            platform_type=platform_type,
            source_file=source_file,
        )

        return profile

    def _get_wmo_number(self, ds: xr.Dataset) -> str:
        """Extract WMO number from dataset."""
        for var_name in ["PLATFORM_NUMBER", "platform_number", "WMO_NUMBER"]:
            if var_name in ds.variables:
                value = ds[var_name].values
                if isinstance(value, np.ndarray):
                    value = value.flat[0]
                if isinstance(value, bytes):
                    return value.decode("utf-8").strip()
                return str(value).strip()
        return "UNKNOWN"

    def _get_timestamp(self, ds: xr.Dataset, prof_idx: int) -> Optional[datetime]:
        """Extract timestamp for profile."""
        for var_name in ["JULD", "juld", "TIME", "time"]:
            if var_name in ds.variables:
                try:
                    time_val = ds[var_name].values
                    if len(time_val.shape) > 0:
                        time_val = time_val[prof_idx]
                    return pd.to_datetime(time_val).to_pydatetime()
                except Exception:
                    continue
        return None

    def _get_value(self, ds: xr.Dataset, var_name: str, prof_idx: int, default=None):
        """Get a scalar value from dataset."""
        if var_name in ds.variables:
            try:
                value = ds[var_name].values
                if len(value.shape) > 0:
                    value = value[prof_idx]
                if isinstance(value, (np.floating, float)) and np.isnan(value):
                    return default
                return value
            except Exception:
                pass
        return default

    def _get_variable_array(self, ds: xr.Dataset, var_type: str, prof_idx: int) -> Optional[List[float]]:
        """Get variable array (e.g., temperature profile)."""
        possible_names = self.VAR_MAPPINGS.get(var_type, [])

        for var_name in possible_names:
            if var_name in ds.variables:
                try:
                    data = ds[var_name].values
                    if len(data.shape) == 2: 
                        data = data[prof_idx, :]
                    result = []
                    for val in data:
                        if isinstance(val, (np.floating, float)) and (np.isnan(val) or val > 1e10):
                            result.append(None)
                        else:
                            result.append(float(val))
                    return result
                except Exception:
                    continue
        return None

    def _get_qc_array(self, ds: xr.Dataset, var_type: str, prof_idx: int) -> Optional[List[int]]:
        """Get QC flags array."""
        possible_names = self.VAR_MAPPINGS.get(var_type, [])
        qc_names = [f"{name}_QC" for name in possible_names]

        for var_name in qc_names:
            if var_name in ds.variables:
                try:
                    data = ds[var_name].values
                    if len(data.shape) == 2:
                        data = data[prof_idx, :]
                    result = []
                    for val in data:
                        try:
                            if isinstance(val, bytes):
                                result.append(int(val.decode("utf-8")))
                            else:
                                result.append(int(val))
                        except Exception:
                            result.append(9) 
                    return result
                except Exception:
                    continue
        return None

    def _get_platform_type(self, ds: xr.Dataset) -> Optional[str]:
        """Get platform type."""
        for var_name in ["PLATFORM_TYPE", "platform_type"]:
            if var_name in ds.variables:
                value = ds[var_name].values
                if isinstance(value, np.ndarray):
                    value = value.flat[0]
                if isinstance(value, bytes):
                    return value.decode("utf-8").strip()
                return str(value).strip()
        return None
