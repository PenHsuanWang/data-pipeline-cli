from enum import Enum
from typing import List, Optional, Any
from pydantic import BaseModel, Field

class HealthStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"

class ConnectionHealth(BaseModel):
    db_alias: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    error_message: Optional[str] = None

class ColumnDef(BaseModel):
    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool = False

class TableSchema(BaseModel):
    table_name: str
    columns: List[ColumnDef]
    row_count: Optional[int] = None

class ConnectionConfig(BaseModel):
    """
    Configuration model for database connection.
    Depending on implementation, connection_string might be constructed
    from these fields or provided directly.
    """
    alias: str
    type: str  # e.g., "oracle", "postgres"
    host: str
    port: int
    database: Optional[str] = None
    service_name: Optional[str] = None  # For Oracle
    username: str
    password: str  # Should be handled securely
    
    @property
    def connection_string(self) -> str:
        # This is a placeholder; actual construction logic might reside in factory
        # or be more complex depending on the driver.
        # Ideally, we return a SQLAlchemy-compatible URL or DSN.
        if self.type == "postgres":
             return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.type == "oracle":
             # Implementing basic thin mode string construction
             dsn = f"{self.host}:{self.port}/{self.service_name}"
             return f"oracle+oracledb://{self.username}:{self.password}@{dsn}"
        return ""

class QualityIssue(BaseModel):
    """Record of a single quality issue"""
    rule_name: str
    column_name: Optional[str] = None
    issue_type: str
    row_indices: List[int] = Field(default_factory=list)
    sample_values: List[str] = Field(default_factory=list)
    count: int
    percentage: float

class QualityReport(BaseModel):
    """Full Quality Report"""
    source_info: dict
    total_rows: int
    issues: List[QualityIssue]
    summary: dict  # Error rate summary per column
