from urllib.parse import urlparse
from typing import Union
from ..domain.interfaces import DatabaseConnector
from ..config import DatabaseConfig
from .postgres import PostgresConnector
from .oracle import OracleConnector
from .base import SQLAlchemyConnector

# Global flag to ensure Oracle Client is initialized only once
_ORACLE_CLIENT_INITIALIZED = False

def _init_oracle_client(lib_dir: str = None):
    """
    Helper to initialize Oracle Instant Client for Thick Mode.
    Ensures it's called only once per process.
    """
    global _ORACLE_CLIENT_INITIALIZED
    if _ORACLE_CLIENT_INITIALIZED:
        return

    try:
        import oracledb
        oracledb.init_oracle_client(lib_dir=lib_dir)
        _ORACLE_CLIENT_INITIALIZED = True
    except ImportError:
        pass  # oracledb not installed, let the connector handle the error later
    except Exception as e:
        # Log warning but don't crash yet; let the connection attempt fail if critical
        print(f"Warning: Failed to initialize Oracle Client: {e}")

def get_connector(config: Union[str, DatabaseConfig], alias: str = "unknown") -> DatabaseConnector:
    """
    Factory function to create the appropriate connector instance.
    Accepts either a connection string (str) or a DatabaseConfig object.
    """
    if isinstance(config, DatabaseConfig):
        connection_string = config.connection_string
        alias = config.alias
        
        # Handle Oracle Thick Mode Initialization
        if config.type == "oracle" and config.oracle_thick_mode:
            _init_oracle_client(config.oracle_lib_dir)
            
    else:
        connection_string = config
    
    # Simple logic: check prefix. 
    if connection_string.startswith("postgresql") or connection_string.startswith("postgres"):
        return PostgresConnector(connection_string, alias)
    elif "oracle" in connection_string:
        # If we have a config object and it's oracle, we might want to pass more info
        # But currently OracleConnector only takes conn string.
        # Ideally, we should update OracleConnector to take the config object too.
        # For now, the global init handled above covers the critical part.
        return OracleConnector(connection_string, alias)
    elif "sqlite" in connection_string:
         return SQLAlchemyConnector(connection_string, alias)
    else:
        # Default fallback to SQLAlchemy generic
        return SQLAlchemyConnector(connection_string, alias)
