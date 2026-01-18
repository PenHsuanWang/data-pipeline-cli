from .base import SQLAlchemyConnector

class PostgresConnector(SQLAlchemyConnector):
    """
    PostgreSQL specific implementation.
    Inherits from Generic SQLAlchemy connector but can override methods
    if specific optimizations (like COPY command) are needed.
    """
    pass
