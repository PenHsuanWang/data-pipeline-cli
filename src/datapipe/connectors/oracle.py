from typing import Optional
from sqlalchemy import create_engine
import sys
from .base import SQLAlchemyConnector

# Global flag to track Oracle Client initialization state within this module
_ORACLE_CLIENT_INITIALIZED = False

def _ensure_oracle_client_init(lib_dir: Optional[str] = None):
    """
    Ensures oracledb.init_oracle_client is called exactly once if needed.
    """
    global _ORACLE_CLIENT_INITIALIZED
    if _ORACLE_CLIENT_INITIALIZED:
        return

    try:
        import oracledb
        oracledb.init_oracle_client(lib_dir=lib_dir)
        _ORACLE_CLIENT_INITIALIZED = True
    except ImportError:
        print("Error: python-oracledb is not installed.", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to initialize Oracle Client: {e}", file=sys.stderr)

class OracleConnector(SQLAlchemyConnector):
    """
    Oracle specific implementation.
    Supports THIN, THICK, and AUTO modes via DatabaseConfig.
    """
    def __init__(self, connection_string: str, db_alias: str = "unknown", config = None):
        super().__init__(connection_string, db_alias)
        self.config = config
        self._determine_mode()

    def _determine_mode(self):
        """
        Parses config to determine the target thick_mode for create_engine.
        """
        # Default to None (Auto behavior in sqlalchemy/oracledb)
        self.thick_mode_arg = None
        
        if not self.config:
            return

        # Circular import prevention: Importing here or assuming config structure is known
        # We access attributes dynamically to be safe or rely on the object passed
        mode = getattr(self.config, 'oracle_client_mode', 'thin') # Default to thin if not present
        lib_dir = getattr(self.config, 'oracle_lib_dir', None)

        # Convert Enum string to lowercase for comparison if needed, though Enum usually handles it
        # Assuming config.oracle_client_mode is the Enum or string value
        mode_val = str(mode).lower()

        if mode_val == 'thick':
            _ensure_oracle_client_init(lib_dir)
            self.thick_mode_arg = True
        elif mode_val == 'thin':
            self.thick_mode_arg = False
        elif mode_val == 'auto':
            # In Auto mode, we try to init if lib_dir is provided, otherwise we let driver decide
            if lib_dir:
                _ensure_oracle_client_init(lib_dir)
            self.thick_mode_arg = None # Let driver/sqlalchemy decide based on global state

    def connect(self) -> None:
        if not self._engine:
            try:
                # Prepare arguments for create_engine
                kwargs = {}
                if self.thick_mode_arg is not None:
                    kwargs['thick_mode'] = self.thick_mode_arg
                
                # Ensure oracle dialect is available
                self._engine = create_engine(self.connection_string, **kwargs)
            except TypeError:
                 # Fallback if thick_mode arg is not supported
                 self._engine = create_engine(self.connection_string)
            except ImportError:
                 print("Error: python-oracledb is not installed. Please install it with `pip install '.[oracle]'`", file=sys.stderr)
                 raise
            except Exception as e:
                raise e