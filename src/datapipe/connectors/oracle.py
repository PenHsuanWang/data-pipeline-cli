from .base import SQLAlchemyConnector
from sqlalchemy import create_engine
import sys

class OracleConnector(SQLAlchemyConnector):
    """
    Oracle specific implementation.
    Ensures Thin Mode is used (python-oracledb).
    """
    def connect(self) -> None:
        if not self._engine:
            try:
                # Ensure oracle dialect is available and thin mode implied by driver
                # python-oracledb is the new default for 'oracle+oracledb://'
                self._engine = create_engine(self.connection_string, thick_mode=False)
            except TypeError:
                 # Fallback if thick_mode arg is not supported by this version of sqlalchemy/dialect
                 self._engine = create_engine(self.connection_string)
            except ImportError:
                 print("Error: python-oracledb is not installed. Please install it with `pip install '.[oracle]'`", file=sys.stderr)
                 raise
            except Exception as e:
                raise e