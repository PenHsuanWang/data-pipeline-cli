from typing import Iterator, List, Optional
import pandas as pd

class Transformer:
    """Basic Transformation Logic"""
    
    def apply(self, data_iter: Iterator[pd.DataFrame], drop_cols: Optional[List[str]] = None, rename_map: Optional[dict] = None) -> Iterator[pd.DataFrame]:
        for df in data_iter:
            # 1. Drop Columns
            if drop_cols:
                # Only drop columns that exist
                existing_cols = [c for c in drop_cols if c in df.columns]
                if existing_cols:
                    df = df.drop(columns=existing_cols)
            
            # 2. Rename Columns
            if rename_map:
                df = df.rename(columns=rename_map)
            
            yield df
