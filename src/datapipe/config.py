from typing import List, Optional
from pathlib import Path
import yaml
from pydantic_settings import BaseSettings
from pydantic import BaseModel, ValidationError
from .exceptions import ConfigurationError

class DatabaseConfig(BaseModel):
    alias: str
    type: str  # oracle, postgres, sqlite, etc.
    connection_string: str
    
    # Oracle Specific Options
    oracle_thick_mode: bool = False
    oracle_lib_dir: Optional[str] = None

class AppConfig(BaseSettings):
    databases: List[DatabaseConfig] = []
    chunk_size: int = 50000

    @classmethod
    def from_yaml(cls, config_path: Path) -> "AppConfig":
        if not config_path.exists():
            raise ConfigurationError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, "r") as f:
                raw_config = yaml.safe_load(f)
            return cls(**raw_config)
        except (ValidationError, yaml.YAMLError) as e:
            raise ConfigurationError(f"Invalid configuration format: {e}")

    def get_db_config(self, alias: str) -> DatabaseConfig:
        for db in self.databases:
            if db.alias == alias:
                return db
        raise ConfigurationError(f"Database alias '{alias}' not found in config")
