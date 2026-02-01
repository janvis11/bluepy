"""Setup script for BluePy."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bluepy",
    version="1.0.0",
    author="BluePy Contributors",
    description="AI Conversational Interface for ARGO Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bluepy",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Oceanography",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "streamlit>=1.31.0",
        "sqlalchemy>=2.0.25",
        "psycopg2-binary>=2.9.9",
        "geoalchemy2>=0.14.3",
        "xarray>=2024.1.0",
        "netCDF4>=1.6.5",
        "pandas>=2.2.0",
        "numpy>=1.26.3",
        "pyarrow>=15.0.0",
        "chromadb>=0.4.22",
        "openai>=1.10.0",
        "plotly>=5.18.0",
        "folium>=0.15.1",
        "streamlit-folium>=0.16.0",
        "python-dotenv>=1.0.1",
        "loguru>=0.7.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.4",
            "pytest-asyncio>=0.23.3",
            "pytest-cov>=4.1.0",
            "black>=24.1.1",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "isort>=5.13.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "bluepy-init-db=scripts.init_db:main",
            "bluepy-ingest=ingestion.pipeline:main",
            "bluepy-embed=scripts.embed_profiles:main",
            "bluepy-sample-data=scripts.sample_data_generator:main",
        ],
    },
)
