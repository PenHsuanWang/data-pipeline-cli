from typing import Iterator
from pathlib import Path
import pandas as pd
from ..domain.interfaces import DatabaseConnector, DataReader

class FileReader:
    """
    Read local files (CSV/Parquet)
    """
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_type = file_path.suffix.lower()
        if not self.file_path.exists():
             raise FileNotFoundError(f"File not found: {self.file_path}")
    
    def read(self, chunk_size: int = 50000) -> Iterator[pd.DataFrame]:
        if self.file_type == ".csv":
            yield from pd.read_csv(self.file_path, chunksize=chunk_size)
        elif self.file_type == ".parquet":
            # Parquet file reading with pyarrow
            # For simplicity, if file is huge, we might need BatchedFileReader from pyarrow
            # Pandas read_parquet doesn't support chunksize directly like read_csv
            # We will use pyarrow.parquet.ParquetFile for streaming
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(self.file_path)
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                yield batch.to_pandas()
        else:
             raise ValueError(f"Unsupported file type: {self.file_type}")

    def get_source_info(self) -> dict:
        return {"type": "file", "path": str(self.file_path)}

class DatabaseReader:
    """Database Reader"""
    def __init__(self, connector: DatabaseConnector, table_name: str):
        self.connector = connector
        self.table_name = table_name
    
    def read(self, chunk_size: int = 50000) -> Iterator[pd.DataFrame]:
        query = f"SELECT * FROM {self.table_name}"
        yield from self.connector.fetch_data(query, chunk_size)

    def get_source_info(self) -> dict:
        return {"type": "database", "table": self.table_name}
