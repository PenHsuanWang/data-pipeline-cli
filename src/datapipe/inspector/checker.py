from ..domain.interfaces import DatabaseConnector
from ..domain.models import ConnectionHealth

class ConnectionChecker:
    """
    SRP: Responsible only for connectivity checks.
    """
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector

    def check_health(self) -> ConnectionHealth:
        return self.connector.check_health()
