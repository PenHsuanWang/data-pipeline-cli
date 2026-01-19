from urllib.parse import urlparse
from typing import Union
from ..domain.interfaces import DatabaseConnector
from ..config import DatabaseConfig
from .postgres import PostgresConnector
from .oracle import OracleConnector
from .base import SQLAlchemyConnector

def get_connector(config: Union[str, DatabaseConfig], alias: str = "unknown") -> DatabaseConnector:
    """
    Factory function to create the appropriate connector instance.
    Accepts either a connection string (str) or a DatabaseConfig object.
    """
    connection_string = config
    if isinstance(config, DatabaseConfig):
        connection_string = config.connection_string
        alias = config.alias
            
    # Simple logic: check prefix. 
    if connection_string.startswith("postgresql") or connection_string.startswith("postgres"):
        return PostgresConnector(connection_string, alias)
    elif "oracle" in connection_string:
        # Pass the full config object if available, allowing OracleConnector to handle initialization logic
        oracle_config = config if isinstance(config, DatabaseConfig) else None
        return OracleConnector(connection_string, alias, config=oracle_config)
    elif "sqlite" in connection_string:
         return SQLAlchemyConnector(connection_string, alias)
    else:
        # Default fallback to SQLAlchemy generic
        return SQLAlchemyConnector(connection_string, alias)
