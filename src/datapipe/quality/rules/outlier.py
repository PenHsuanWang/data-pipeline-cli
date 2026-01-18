from typing import List
import pandas as pd
import numpy as np
from ...domain.interfaces import QualityRule
from ...domain.models import QualityIssue

class ZScoreOutlierRule:
    """
    Strategy: Detect outliers using Z-Score.
    Note: This implementation calculates Z-Score per chunk (Local Z-Score).
    For strict global Z-Score, a two-pass approach is required which involves
    pre-calculating global mean/std.
    """
    def __init__(self, threshold: float = 3.0):
        self._name = f"Z-Score Outlier Check (Threshold={threshold})"
        self.threshold = threshold

    @property
    def name(self) -> str:
        return self._name

    def check(self, df: pd.DataFrame) -> List[QualityIssue]:
        issues = []
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Calculate Local Z-Score
            mean = df[col].mean()
            std = df[col].std()
            
            # Avoid division by zero
            if std == 0 or pd.isna(std):
                continue
                
            z_scores = (df[col] - mean) / std
            outliers_mask = z_scores.abs() > self.threshold
            outliers = df[outliers_mask]
            
            count = len(outliers)
            if count > 0:
                percentage = (count / len(df)) * 100
                sample_values = outliers[col].head(3).astype(str).tolist()
                
                issues.append(QualityIssue(
                    rule_name=self.name,
                    column_name=col,
                    issue_type="Outlier",
                    count=count,
                    percentage=round(percentage, 2),
                    sample_values=sample_values
                ))
        return issues
