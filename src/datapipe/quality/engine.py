from typing import List, Dict
import pandas as pd
from ..domain.interfaces import DataReader, QualityRule
from ..domain.models import QualityReport, QualityIssue

class QualityReporter:
    def __init__(self):
        self.issues: List[QualityIssue] = []
        self.total_rows = 0
        self.source_info = {}

    def add(self, new_issues: List[QualityIssue]):
        self.issues.extend(new_issues)

    def generate(self) -> QualityReport:
        # Aggregation logic to merge chunked issues
        # Simple implementation: just list them all
        # Production: Group by column and rule, sum counts, re-calculate weighted percentages
        
        # Simple aggregation for 'Missing Value'
        aggregated_issues = {}
        
        for issue in self.issues:
            key = (issue.rule_name, issue.column_name)
            if key not in aggregated_issues:
                aggregated_issues[key] = issue
            else:
                # Merge logic
                existing = aggregated_issues[key]
                existing.count += issue.count
                # Percentage needs to be recalculated against TOTAL rows, handled below
        
        final_issues = list(aggregated_issues.values())
        
        # Recalculate global percentage
        if self.total_rows > 0:
            for issue in final_issues:
                issue.percentage = round((issue.count / self.total_rows) * 100, 2)
        
        # Generate Summary
        summary = {}
        for issue in final_issues:
            if issue.column_name not in summary:
                summary[issue.column_name] = {}
            summary[issue.column_name][issue.issue_type] = f"{issue.percentage}%"

        return QualityReport(
            source_info=self.source_info,
            total_rows=self.total_rows,
            issues=final_issues,
            summary=summary
        )

class QualityEngine:
    def __init__(self, rules: List[QualityRule]):
        self.rules = rules
        self.reporter = QualityReporter()
    
    def run(self, reader: DataReader, chunk_size: int = 50000) -> QualityReport:
        self.reporter.source_info = reader.get_source_info()
        
        for chunk in reader.read(chunk_size):
            self.reporter.total_rows += len(chunk)
            for rule in self.rules:
                issues = rule.check(chunk)
                self.reporter.add(issues)
                
        return self.reporter.generate()
    
    def run_from_file(self, file_path: str) -> QualityReport:
        from pathlib import Path
        reader = FileReader(Path(file_path))
        return self.run(reader)
    
    # run_from_db needs to inject config externally or handle it in main
