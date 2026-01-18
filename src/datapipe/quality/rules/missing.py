from typing import List
import pandas as pd
from ...domain.interfaces import QualityRule
from ...domain.models import QualityIssue

class MissingValueRule:
    """
    Strategy: Check for Null/NaN values.
    """
    def __init__(self):
        self._name = "Missing Value Check"

    @property
    def name(self) -> str:
        return self._name

    def check(self, df: pd.DataFrame) -> List[QualityIssue]:
        issues = []
        for col in df.columns:
            # Count nulls
            null_count = df[col].isnull().sum()
            if null_count > 0:
                # Find indices (relative to this chunk)
                # In real prod, we might need absolute indices if we track row offset
                # For now, we just report count
                percentage = (null_count / len(df)) * 100
                
                issues.append(QualityIssue(
                    rule_name=self.name,
                    column_name=col,
                    issue_type="Missing Value",
                    count=int(null_count),
                    percentage=round(percentage, 2),
                    sample_values=[] # Can fill with sample if needed
                ))
        return issues
