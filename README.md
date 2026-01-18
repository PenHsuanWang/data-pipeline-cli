# Data Pipeline CLI Tool: Manufacturing Data Integration & Quality Toolkit

A modular CLI tool for database connectivity checks, data quality inspection, and ETL tasks.

## Installation

```bash
pip install -e .
```

## Usage

### 1. Check Connectivity
```bash
datapipe check-conn --config config.yaml
```

### 2. Inspect Data Quality
```bash
# From File
datapipe inspect --source ./data.csv

# From Database
datapipe inspect --db-target pg_prod/users --config config.yaml
```

### 3. Export Data
```bash
datapipe export --input ./data.csv --to-parquet ./output.parquet
```
