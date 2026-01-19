# PR: Oracle Hybrid Connectivity Architecture & Security Hardening (v2)

## 📋 Summary

This PR implements the **Oracle Hybrid Client Architecture** to address the needs of manufacturing environments with heterogeneous databases (coexistence of modern and legacy Oracle versions). It also hardens security defenses at the application level.

Core changes include:
1.  **Oracle Thick Mode Support**: Solves connectivity issues with legacy databases (Oracle 10g/11g) or environments requiring NNE encryption.
2.  **Dynamic Connector Configuration**: Refactored the internal connector instantiation logic to ensure configuration flags (like Thick Mode) are correctly propagated to the SQLAlchemy engine.
3.  **Security Upgrades**: Established "Application-Level Write Blocking" standards in the PRD and recommended the use of Read-Only accounts.

---

## 🚀 Key Changes

### 1. Core Feature: Oracle Thick Mode Support
To support legacy Oracle Database versions on the factory floor, we implemented a switching mechanism for Thick Client based on `python-oracledb`.

*   **Config Extension**: Added `oracle_thick_mode` (bool) and `oracle_lib_dir` (path) parameters to `DatabaseConfig`.
*   **Factory Refactoring**:
    *   Implemented global one-time initialization logic (`_init_oracle_client`) in `src/datapipe/connectors/factory.py` to prevent program crashes caused by repeated calls to `init_oracle_client`.
    *   Updated `get_connector` to extract `oracle_thick_mode` from the config and pass it into the `OracleConnector` constructor.
*   **Connector Update**:
    *   Refactored `OracleConnector` in `src/datapipe/connectors/oracle.py` to accept a `thick_mode` argument.
    *   The `connect()` method now dynamically invokes `create_engine(..., thick_mode=self.thick_mode)`, ensuring the driver mode matches the user's configuration (fixing a previous issue where Thin Mode was hardcoded).

### 2. Architectural Adjustments
*   **Main Entry Point**: Updated the `check_conn`, `inspect`, and `export` commands in `main.py` to ensure advanced parameters from the config file are correctly passed to the Connector layer via the Factory.
*   **PRD Update**: Synchronized updates to `PRD_修正與功能整合.md`, incorporating the complete "Oracle Connectivity Strategy" and "Security Standards".

---

## 🛠 Technical Details

### Global Initialization Pattern
Since `oracledb.init_oracle_client()` can only be executed once within a single Process lifecycle, we adopted the following pattern in `factory.py`:

```python
# src/datapipe/connectors/factory.py

_ORACLE_CLIENT_INITIALIZED = False

def _init_oracle_client(lib_dir: str = None):
    global _ORACLE_CLIENT_INITIALIZED
    if _ORACLE_CLIENT_INITIALIZED:
        return
    # ... Execute initialization ...
    _ORACLE_CLIENT_INITIALIZED = True
```

### Connector Parameter Propagation
To ensure the SQLAlchemy engine respects the configuration, the parameter flow is now:

1.  **Config (`config.yaml`)**: User sets `oracle_thick_mode: true`.
2.  **Factory (`factory.py`)**: `get_connector` reads this boolean and initializes `OracleConnector(..., thick_mode=True)`.
3.  **Connector (`oracle.py`)**:
    ```python
    def connect(self) -> None:
        # ...
        self._engine = create_engine(self.connection_string, thick_mode=self.thick_mode)
    ```

### Configuration Example

Developers can now configure flexibly in `config.yaml`:

```yaml
databases:
  # Existing: Use default Thin Mode (Suitable for ADB, 12c+)
  - alias: "cloud_db"
    type: "oracle"
    connection_string: "..."

  # New: Use Thick Mode (Suitable for Legacy 11g, NNE)
  - alias: "legacy_factory_db"
    type: "oracle"
    connection_string: "..."
    oracle_thick_mode: true
    oracle_lib_dir: "/opt/oracle/instantclient_19_8"  # Optional
```

---

## ✅ Test Plan

1.  **Regression Testing**:
    *   Run `datapipe check-conn` against existing Thin Mode databases to verify that connection and Schema parsing functions remain normal.
2.  **Feature Testing**:
    *   Prepare a configuration file enabling `oracle_thick_mode: true`.
    *   Run `datapipe check-conn` and observe logs to see if the Oracle Client is correctly initialized.
    *   Verify that `OracleConnector` is indeed using Thick Mode by checking connection properties or forcing a legacy connection (11g) which would fail in Thin Mode.

---

## 📄 Documentation
*   [Updated PRD](PRD_修正與功能整合.md)

---

**Reviewers**: Please verify that `factory.py` correctly passes the boolean flag and that `oracle.py` uses `self.thick_mode` in the `create_engine` call.
