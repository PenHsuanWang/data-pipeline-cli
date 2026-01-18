from typing import Iterator, List
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from ..domain.interfaces import DatabaseConnector

class Extractor:
    """Data Extractor supporting multiple sources"""
    
    def from_file(self, file_path: Path, chunk_size: int = 50000) -> Iterator[pd.DataFrame]:
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            yield from pd.read_csv(file_path, chunksize=chunk_size)
        elif suffix == ".parquet":
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                yield batch.to_pandas()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def from_db(self, connector: DatabaseConnector, query: str, chunk_size: int = 50000) -> Iterator[pd.DataFrame]:
        yield from connector.fetch_data(query, chunk_size)
