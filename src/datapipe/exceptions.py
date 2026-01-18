class DatapipeException(Exception):
    """Base Exception Class"""
    pass

class ConnectionError(DatapipeException):
    """Connection Failure"""
    pass

class ConfigurationError(DatapipeException):
    """Configuration Error"""
    pass

class DataSourceError(DatapipeException):
    """Data Source Error (File not found, malformed, etc.)"""
    pass
