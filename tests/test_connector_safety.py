import pytest
from sqlalchemy import create_engine, text
from datapipe.connectors.base import SQLAlchemyConnector

# Mocking the connector to use SQLite for testing safety mechanisms
class TestSafetyConnector(SQLAlchemyConnector):
    def __init__(self):
        # Use in-memory SQLite for fast testing
        super().__init__("sqlite:///:memory:", "test_db")

def test_read_only_listener_allows_select():
    """Test that SELECT statements are allowed."""
    connector = TestSafetyConnector()
    connector.connect()
    
    try:
        with connector._engine.connect() as conn:
            # Create a dummy table bypassing the listener for setup if possible, 
            # or just test SELECT 1 which doesn't need a table.
            # Since our listener blocks CREATE, we can't easily create a table 
            # using the protected engine.
            # But we can test that SELECT 1 works.
            result = conn.execute(text("SELECT 1")).scalar()
            assert result == 1
    except PermissionError:
        pytest.fail("Valid SELECT statement was blocked.")
    finally:
        connector.close()

def test_read_only_listener_blocks_create():
    """Test that CREATE TABLE statements are blocked."""
    connector = TestSafetyConnector()
    connector.connect()
    
    with pytest.raises(PermissionError) as excinfo:
        with connector._engine.connect() as conn:
            conn.execute(text("CREATE TABLE test (id int)"))
    
    assert "SAFETY BLOCK" in str(excinfo.value)
    connector.close()

def test_read_only_listener_blocks_drop():
    """Test that DROP TABLE statements are blocked."""
    connector = TestSafetyConnector()
    connector.connect()
    
    with pytest.raises(PermissionError) as excinfo:
        with connector._engine.connect() as conn:
            conn.execute(text("DROP TABLE users"))
    
    assert "SAFETY BLOCK" in str(excinfo.value)
    connector.close()

def test_read_only_listener_blocks_insert():
    """Test that INSERT statements are blocked."""
    connector = TestSafetyConnector()
    connector.connect()
    
    with pytest.raises(PermissionError) as excinfo:
        with connector._engine.connect() as conn:
            conn.execute(text("INSERT INTO users VALUES (1)"))
    
    assert "SAFETY BLOCK" in str(excinfo.value)
    connector.close()

def test_read_only_listener_blocks_update():
    """Test that UPDATE statements are blocked."""
    connector = TestSafetyConnector()
    connector.connect()
    
    with pytest.raises(PermissionError) as excinfo:
        with connector._engine.connect() as conn:
            conn.execute(text("UPDATE users SET name='admin'"))
    
    assert "SAFETY BLOCK" in str(excinfo.value)
    connector.close()

def test_read_only_listener_allows_explain():
    """Test that EXPLAIN statements are allowed."""
    connector = TestSafetyConnector()
    connector.connect()
    
    try:
        with connector._engine.connect() as conn:
            # SQLite supports EXPLAIN QUERY PLAN
            conn.execute(text("EXPLAIN QUERY PLAN SELECT 1"))
    except PermissionError:
        pytest.fail("Valid EXPLAIN statement was blocked.")
    finally:
        connector.close()

def test_read_only_listener_allows_with_cte():
    """Test that WITH (CTE) statements are allowed."""
    connector = TestSafetyConnector()
    connector.connect()
    
    try:
        with connector._engine.connect() as conn:
            conn.execute(text("WITH t AS (SELECT 1 as a) SELECT a FROM t"))
    except PermissionError:
        pytest.fail("Valid WITH (CTE) statement was blocked.")
    finally:
        connector.close()
