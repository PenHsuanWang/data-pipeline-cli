from typing import List, Optional
from pydantic import BaseModel
from ..domain.interfaces import DatabaseConnector
from ..domain.models import ConnectionHealth, TableSchema
from .checker import ConnectionChecker
from .crawler import SchemaCrawler

class InspectionReport(BaseModel):
    health: ConnectionHealth
    schema: Optional[dict[str, TableSchema]] = None

class InspectorFacade:
    """
    Facade Pattern: Unified entry point for F1 functionality.
    """
    def __init__(self, connector: DatabaseConnector):
        self._checker = ConnectionChecker(connector)
        self._crawler = SchemaCrawler(connector)

    def run_diagnostics(self) -> InspectionReport:
        # 1. Check connection first (Fail Fast)
        health = self._checker.check_health()
        if health.status != "success":
            return InspectionReport(health=health, schema=None)
        
        # 2. Crawl Schema only if connection is successful
        schema = self._crawler.extract_all()
        return InspectionReport(health=health, schema=schema)
