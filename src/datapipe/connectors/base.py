from typing import Iterator, List, Any, Optional
import time
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from ..domain.interfaces import DatabaseConnector
from ..domain.models import TableSchema, ColumnDef, HealthStatus, ConnectionHealth
from ..exceptions import ConnectionError

class SQLAlchemyConnector:
    """
    Generic SQLAlchemy Connector that can be specialized for Postgres/Oracle
    if needed, or used directly for compliant dialects.
    """
    def __init__(self, connection_string: str, db_alias: str = "unknown"):
        self.connection_string = connection_string
        self.db_alias = db_alias
        self._engine = None

    def connect(self) -> None:
        if not self._engine:
            try:
                # Create engine but don't connect yet (lazy)
                self._engine = create_engine(self.connection_string)
            except Exception as e:
                raise ConnectionError(f"Failed to create engine: {e}")

    def close(self) -> None:
        if self._engine:
            self._engine.dispose()
            self._engine = None

    def check_health(self) -> ConnectionHealth:
        self.connect()
        start_time = time.time()
        status = HealthStatus.FAILED
        error_msg = None
        
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                status = HealthStatus.SUCCESS
        except Exception as e:
            error_msg = str(e)
            status = HealthStatus.FAILED
        
        latency = (time.time() - start_time) * 1000  # ms
        
        if latency > 5000: # 5s timeout simulation for reporting
             # In real-world, connection timeout is handled by driver params,
             # this checks mostly for slow responses if connection succeeded.
             if status == HealthStatus.SUCCESS:
                 status = HealthStatus.TIMEOUT

        return ConnectionHealth(
            db_alias=self.db_alias,
            status=status,
            latency_ms=round(latency, 2),
            error_message=error_msg
        )

    def get_schema(self, table_name: str) -> TableSchema:
        self.connect()
        try:
            inspector = inspect(self._engine)
            columns_info = inspector.get_columns(table_name)
            pk_constraint = inspector.get_pk_constraint(table_name)
            pk_cols = pk_constraint.get('constrained_columns', [])

            if not columns_info:
                 # Check if table exists or permission issue
                 raise ConnectionError(f"Table '{table_name}' not found or not accessible.")

            cols = []
            for col in columns_info:
                cols.append(ColumnDef(
                    name=col['name'],
                    data_type=str(col['type']),
                    is_nullable=col['nullable'],
                    is_primary_key=col['name'] in pk_cols
                ))
            
            # Simple row count (optional, can be expensive on big tables)
            # Keeping it lightweight for now or using distinct query
            return TableSchema(
                table_name=table_name,
                columns=cols,
                row_count=None 
            )
        except SQLAlchemyError as e:
            raise ConnectionError(f"Failed to get schema for {table_name}: {e}")

    def get_all_tables(self) -> List[str]:
        self.connect()
        try:
            inspector = inspect(self._engine)
            return inspector.get_table_names()
        except SQLAlchemyError as e:
            raise ConnectionError(f"Failed to list tables: {e}")

    def fetch_data(self, query: str, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        self.connect()
        try:
            # stream_results=True is critical for server-side cursors in psycopg
            with self._engine.connect().execution_options(stream_results=True) as conn:
                for chunk in pd.read_sql(
                    text(query), 
                    conn, 
                    chunksize=chunk_size
                ):
                    yield chunk
        except SQLAlchemyError as e:
            raise ConnectionError(f"Failed to fetch data: {e}")
