from typing import Dict, List, Optional
from ..domain.interfaces import DatabaseConnector
from ..domain.models import TableSchema

class SchemaCrawler:
    """
    SRP: Responsible only for metadata/schema crawling.
    """
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector

    def extract_all(self) -> Dict[str, TableSchema]:
        """
        Crawls all visible tables and their schemas.
        """
        schemas = {}
        try:
            tables = self.connector.get_all_tables()
            for table in tables:
                # In a real scenario, might want to parallelize this or limit scope
                try:
                    schema = self.connector.get_schema(table)
                    schemas[table] = schema
                except Exception:
                    # Log error but continue for other tables
                    continue
        except Exception:
             # If listing tables fails, return empty
             pass
        return schemas
