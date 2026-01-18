from urllib.parse import urlparse
from ..domain.interfaces import DatabaseConnector
from .postgres import PostgresConnector
from .oracle import OracleConnector
from .base import SQLAlchemyConnector

def get_connector(connection_string: str, alias: str = "unknown") -> DatabaseConnector:
    """
    Factory function to create the appropriate connector instance.
    """
    # Simple logic: check prefix. 
    # Real-world: might use regex or urlparse scheme.
    if connection_string.startswith("postgresql") or connection_string.startswith("postgres"):
        return PostgresConnector(connection_string, alias)
    elif "oracle" in connection_string:
        return OracleConnector(connection_string, alias)
    elif "sqlite" in connection_string:
         return SQLAlchemyConnector(connection_string, alias)
    else:
        # Default fallback to SQLAlchemy generic
        return SQLAlchemyConnector(connection_string, alias)
