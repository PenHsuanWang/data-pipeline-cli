from .base import SQLAlchemyConnector
from sqlalchemy import create_engine
import sys

class OracleConnector(SQLAlchemyConnector):
    """
    Oracle specific implementation.
    Supports both Thin Mode (default) and Thick Mode (if configured).
    """
    def __init__(self, connection_string: str, db_alias: str = "unknown", thick_mode: bool = False):
        super().__init__(connection_string, db_alias)
        self.thick_mode = thick_mode

    def connect(self) -> None:
        if not self._engine:
            try:
                # Ensure oracle dialect is available
                # python-oracledb is the new default for 'oracle+oracledb://'
                # thick_mode=True/False controls the driver mode if supported by the dialect
                self._engine = create_engine(self.connection_string, thick_mode=self.thick_mode)
            except TypeError:
                 # Fallback if thick_mode arg is not supported by this version of sqlalchemy/dialect
                 self._engine = create_engine(self.connection_string)
            except ImportError:
                 print("Error: python-oracledb is not installed. Please install it with `pip install '.[oracle]'`", file=sys.stderr)
                 raise
            except Exception as e:
                raise e