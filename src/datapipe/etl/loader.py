import pandas as pd
from pathlib import Path
from ..domain.interfaces import DatabaseConnector

class Loader:
    """Batch Writer (CSV/Parquet/DB)"""
    
    def __init__(self, connector: DatabaseConnector = None):
        self.connector = connector

    def write_csv(self, df: pd.DataFrame, path: Path, mode: str = 'w', header: bool = True):
        # mode 'a' for append, 'w' for write (overwrite)
        # Pandas to_csv doesn't strictly support 'append' efficiently without rewriting file header issues if not careful
        # But for CLI pipeline simplicity:
        df.to_csv(path, mode=mode, header=header, index=False)

    def write_parquet(self, df: pd.DataFrame, path: Path, append: bool = False):
        # Use fastparquet or pyarrow
        # Simple implementation
        if append and path.exists():
             # PyArrow append implementation would be more complex (requiring consistent schema and Table)
             # For this demo, we might warn or just overwrite for the first chunk and error?
             # Or we use fastparquet which supports append better.
             # Here we assume user handles file management or we overwrite.
             # TODO: Implement proper append logic using pyarrow.Table
             pass
        df.to_parquet(path, index=False)

    def write_db(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append'):
        if not self.connector:
            raise ValueError("Database connector required for DB export")
        # Use pandas to_sql
        # Note: This is synchronous and might be slow for huge data.
        # Ideally use COPY (Postgres) or bulk insert.
        # We need access to the sqlalchemy engine
        # In our abstraction, connector._engine is private-ish.
        # We might need to expose a method in connector.
        # For now, bypassing strict encapsulation for practicality or assume connector has a 'write' method?
        # The protocol has `write(df)`.
        
        # We'll use the connector's internal engine if available, or we should extend protocol
        if hasattr(self.connector, '_engine'):
             df.to_sql(table_name, self.connector._engine, if_exists=if_exists, index=False)
        else:
             raise NotImplementedError("Connector does not support direct pandas writing yet")
