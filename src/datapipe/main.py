import typer
from typing import Optional
from pathlib import Path
from .config import AppConfig
from .connectors.factory import get_connector
from .inspector import InspectorFacade
from .quality.engine import QualityEngine
from .quality.rules.missing import MissingValueRule
from .quality.rules.outlier import ZScoreOutlierRule
from .quality.reader import DatabaseReader
from .etl.pipeline import Pipeline

app = typer.Typer(help="Manufacturing Data Integration & Quality Toolkit")

@app.command()
def check_conn(
    config: Path = typer.Option(..., "--config", "-c", help="Path to configuration file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """
    F1: Connectivity Health Check and Schema Discovery.
    Reads config file, generates connectivity report, and exits.
    """
    try:
        app_config = AppConfig.from_yaml(config)
    except Exception as e:
        typer.echo(f"Error loading config: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Starting connectivity check for {len(app_config.databases)} databases...")
    
    for db_config in app_config.databases:
        typer.echo(f"Checking {db_config.alias} ({db_config.type})...")
        
        try:
            connector = get_connector(db_config, db_config.alias)
            facade = InspectorFacade(connector)
            report = facade.run_diagnostics()
            
            # Output Result
            if report.health.status == "success":
                typer.secho(f"✅ {db_config.alias}: Connection Successful ({report.health.latency_ms}ms)", fg=typer.colors.GREEN)
                if report.schema_info:
                    typer.echo(f"   Found {len(report.schema_info)} tables.")
                    if verbose:
                        for table, schema in report.schema_info.items():
                            typer.echo(f"     - {table}: {len(schema.columns)} columns")
            else:
                 typer.secho(f"❌ {db_config.alias}: Connection Failed. Error: {report.health.error_message}", fg=typer.colors.RED)
                 
        except Exception as e:
             typer.secho(f"⚠️ Error processing {db_config.alias}: {e}", fg=typer.colors.YELLOW)
             if verbose:
                 raise e

@app.command()
def inspect(
    source: Optional[Path] = typer.Option(None, "--source", help="Local file path (CSV/Parquet)"),
    db_target: Optional[str] = typer.Option(None, "--db-target", help="Database/Table (alias/table)"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to configuration file (required for db-target)"),
):
    """
    F2: Data Quality Inspection.
    Supports reading data from databases or local files (CSV/Parquet) for quality inspection.
    """
    # Initialize Engine with Rules
    rules = [MissingValueRule(), ZScoreOutlierRule()] # Add more rules here
    engine = QualityEngine(rules)
    
    report = None
    try:
        if source:
            typer.echo(f"Inspecting local file: {source}")
            report = engine.run_from_file(str(source))
        elif db_target:
            if not config:
                typer.echo("Error: --config is required when using --db-target", err=True)
                raise typer.Exit(code=1)
            
            # Parse Target
            try:
                db_alias, table_name = db_target.split("/")
            except ValueError:
                 typer.echo("Error: --db-target format must be 'alias/table_name'", err=True)
                 raise typer.Exit(code=1)
                 
            # Load Config & Connector
            app_config = AppConfig.from_yaml(config)
            db_config = app_config.get_db_config(db_alias)
            connector = get_connector(db_config, db_config.alias)
            
            # Run
            reader = DatabaseReader(connector, table_name)
            report = engine.run(reader, chunk_size=app_config.chunk_size)
        else:
            typer.echo("Error: Either --source or --db-target must be provided", err=True)
            raise typer.Exit(code=1)
            
        # Output Report
        typer.echo("\n🔍 Data Quality Report")
        typer.echo("=========================")
        typer.echo(f"Total Rows Processed: {report.total_rows}")
        typer.echo("\nIssues Found:")
        if not report.issues:
            typer.secho("✅ No issues found!", fg=typer.colors.GREEN)
        else:
            for issue in report.issues:
                typer.echo(f"  - {issue.rule_name} on '{issue.column_name}': {issue.count} errors ({issue.percentage}%)")
        
    except Exception as e:
        typer.secho(f"❌ Error during inspection: {e}", fg=typer.colors.RED)
        if config: # Very rough check, usually use verbose flag
             pass # raise e

@app.command()
def export(
    input: Path = typer.Option(..., "--input", help="Input file path"),
    to_csv: Optional[Path] = typer.Option(None, "--to-csv", help="Output CSV path"),
    to_parquet: Optional[Path] = typer.Option(None, "--to-parquet", help="Output Parquet path"),
    to_db: Optional[str] = typer.Option(None, "--to-db", help="Target database alias (e.g., 'pg_prod')"),
    target_table: Optional[str] = typer.Option(None, "--target-table", help="Target table name (required if to-db is set)"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to configuration file (required for to-db)"),
):
    """
    F3: Data Integration and Export.
    Supports reading data from local files or memory objects, exporting to CSV or target databases.
    """
    typer.echo(f"Starting export from: {input}")
    
    try:
        pipeline = Pipeline()
        pipeline.extract_from_file(str(input))
        
        # Optional Transformation (e.g., driven by config or CLI args not fully exposed yet)
        pipeline.transform() 
        
        connector = None
        if to_db:
             if not config or not target_table:
                 typer.echo("Error: --config and --target-table are required when using --to-db", err=True)
                 raise typer.Exit(code=1)
             
             app_config = AppConfig.from_yaml(config)
             db_config = app_config.get_db_config(to_db)
             connector = get_connector(db_config, db_config.alias)
             # Important: Connect explicitly if loader needs it, though loader might handle it
             connector.connect() 
        
        # Execute Load
        result = pipeline.load(
            to_csv=str(to_csv) if to_csv else None, 
            to_parquet=str(to_parquet) if to_parquet else None, 
            to_db=target_table if to_db else None,
            connector=connector
        )
        
        typer.secho(f"✅ {result}", fg=typer.colors.GREEN)
        
    except Exception as e:
         typer.secho(f"❌ Export failed: {e}", fg=typer.colors.RED)
         raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
