#%% [markdown]
# # 📊 Feature Profiling for Q6 Dataset
#
# ## Analysis Coverage
# - Column metadata: dtypes, row counts, memory usage
# - Missing values: counts, ratios, top-N most-missing features
# - Numeric features: stats (min/max/mean/std/percentiles), IQR, decile range, outliers (Tukey fences), skewness, kurtosis
# - Categorical features: unique counts, top values and frequency distributions
# - Boolean features: true/false counts and ratios
# - Datetime features: auto-parse from object columns
# - Feature categorization: discrete (categorical/integer/bool) vs continuous with distribution types (small/large spread, skewed)
# - Outputs: CSV master table, plain-text report, Markdown report (can link figures)
#
# ## Usage
# 1. Execute cells in order (or run only the ones you need).
# 2. Adjust input/output paths in the “Parameter Configuration” cell.
# 3. All outputs are saved under `data_profiling_output/`.
#
# > Note: This script has no `main()` or CLI args; it is designed for Notebook/Interactive execution.
#
# References:
# - pandas `describe()`, `value_counts()`, `DataFrame.corr(method="spearman")`
# - matplotlib `pyplot.hist`
# - ydata-profiling (optional)
#
#%% [markdown]
# ## 0) Imports, display settings, and output directories
#
#%%
import io
import os
import datetime
from pathlib import Path
from typing import Any, List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_bool_dtype
import matplotlib.pyplot as plt
import logging
try:
    from IPython.display import display
except Exception:
    display = print

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", lambda v: f"{v:.6g}")

ROOT = Path.cwd()
PROD_DIR = ROOT / "data_profiling_output"
FIGS_DIR = PROD_DIR / "figs"
for d in [PROD_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

#
#%% [markdown]
# ## 1) Logger and helper utilities
#
#%%
def setup_logger(level=logging.INFO):
    logger = logging.getLogger("feature_analysis")
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(level)
    return logger

logger = setup_logger()

def section(title: str):
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)

def subtitle(msg: str):
    logger.info(msg)

def _safe_scalar(x: Any) -> Any:
    if pd.isna(x):
        return np.nan
    try:
        if isinstance(x, (np.generic,)):
            return x.item()
    except Exception:
        pass
    try:
        if isinstance(x, pd.Timestamp):
            return x.isoformat()
    except Exception:
        pass
    return x

def show_interactive_summary(summary: pd.DataFrame, df: Optional[pd.DataFrame] = None):
    total_feats = len(summary)
    cat_counts = summary['feature_category'].value_counts().to_dict() if 'feature_category' in summary.columns else {}
    dist_counts = summary['distribution_type'].value_counts().to_dict() if 'distribution_type' in summary.columns else {}

    section("Feature quick summary")
    logger.info(f"Total features: {total_feats}")
    if df is not None:
        logger.info(f"Rows in dataset: {len(df)}")

    if cat_counts:
        logger.info("Feature categories:")
        for k, v in cat_counts.items():
            logger.info(f"  {k}: {v}")

    if dist_counts:
        logger.info("Distribution types (summary):")
        for k, v in dist_counts.items():
            logger.info(f"  {k}: {v}")

    try:
        examples = {}
        for cat in ['discrete', 'continuous']:
            if 'feature_category' in summary.columns and (summary['feature_category'] == cat).any():
                examples[cat] = summary[summary['feature_category'] == cat]['feature'].head(5).tolist()
        if examples:
            logger.info("Example features:")
            for cat, feats in examples.items():
                logger.info(f"  {cat}: {', '.join(map(str, feats))}")
    except Exception:
        pass

#
#%% [markdown]
# ## 2) Parameter configuration (input & output paths)
#
#%%
CANDIDATES = [Path("../data/Q6.csv"), ROOT / "data" / "Q6.csv"]
DATA_PATH = next((p for p in CANDIDATES if p.exists()), CANDIDATES[-1])

OUT_SUMMARY_CSV = PROD_DIR / "feature_summary_Q6.csv"
OUT_TEXT =     PROD_DIR / "feature_report_Q6.txt"
OUT_MD =       PROD_DIR / "feature_report_Q6.md"

print(f"[INFO] DATA_PATH     = {DATA_PATH}")
print(f"[INFO] OUTPUT (CSV)  = {OUT_SUMMARY_CSV}")
print(f"[INFO] OUTPUT (TXT)  = {OUT_TEXT}")
print(f"[INFO] OUTPUT (MD)   = {OUT_MD}")
print(f"[INFO] FIGS_DIR      = {FIGS_DIR}")

#
#%% [markdown]
# ## 3) Load dataset & initial diagnostics (shape/columns/info)
#
#%%
section("Load dataset & diagnostics")
df = pd.read_csv(DATA_PATH)

logger.info(f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns")
logger.info(f"First 8 columns: {df.columns.tolist()[:8]}")
logger.info(f"Last 8  columns: {df.columns.tolist()[-8:]}")

cols = df.columns.tolist()
dups = [c for c in set(cols) if cols.count(c) > 1]
if dups:
    logger.warning("Duplicate column names detected: %s", dups)

unnamed = [c for c in cols if (isinstance(c, str) and (c.strip() == "" or c.lower().startswith("unnamed")))]
if unnamed:
    logger.warning("Unnamed/empty column headers detected: %s", unnamed)

all_null_cols = [c for c in cols if df[c].isna().all()]
if all_null_cols:
    logger.warning("Columns with all-null values (count=%d): %s", len(all_null_cols), all_null_cols[:20])

subtitle("First 5 rows:")
display(df.head())

subtitle("DataFrame.info():")
_buf = io.StringIO()
df.info(buf=_buf)
print(_buf.getvalue())

#
#%% [markdown]
# ## 4) Core function: per-column summary (`describe_dataframe`)
#
#%%
def describe_dataframe(df: pd.DataFrame, sample_n: Optional[int] = None, infer_numeric_from_strings: bool = False) -> pd.DataFrame:
    rows = []
    n = len(df)
    df_proc = df.copy()

    # Optionally coerce mostly-numeric object columns to numeric
    if infer_numeric_from_strings:
        for c in df_proc.select_dtypes(include=["object"]).columns:
            coerced = pd.to_numeric(df_proc[c], errors="coerce")
            frac_numeric = coerced.notna().sum() / max(1, df_proc[c].notna().sum())
            if frac_numeric >= 0.9:
                df_proc[c] = coerced

    for i, col in enumerate(df_proc.columns):
        ser = df_proc[col]
        dtype = ser.dtype

        non_null = int(ser.notna().sum())
        missing = int(ser.isna().sum())
        missing_ratio = missing / n if n > 0 else np.nan

        try:
            unique_count = int(ser.nunique(dropna=True))
        except Exception:
            unique_count = int(ser.dropna().astype(str).nunique())

        top_values = None
        sample_values = None

        try:
            memory_bytes = int(ser.memory_usage(deep=True))
        except Exception:
            memory_bytes = np.nan

        is_num = bool(is_numeric_dtype(ser))
        is_dt = bool(is_datetime64_any_dtype(ser))
        is_bool = bool(is_bool_dtype(ser))

        min_val = max_val = mean = std = np.nan
        p05 = p10 = p25 = p50 = p75 = p90 = p95 = np.nan
        iqr = np.nan
        decile_range = np.nan
        outlier_lower_count = outlier_upper_count = outlier_count = 0
        outlier_ratio = np.nan
        extreme_outlier_count = 0
        zeros_count = neg_count = pos_count = np.nan
        skew = kurt = np.nan
        true_count = false_count = true_ratio = np.nan

        # Boolean summary
        if is_bool:
            vals_bool = ser.dropna().astype(bool)
            true_count = int(vals_bool.sum())
            false_count = int((~vals_bool).sum())
            true_ratio = float(true_count / max(1, len(vals_bool)))

        # Datetime: use dtype or attempt parsing for object columns
        if is_dt:
            try:
                parsed_dt = pd.to_datetime(ser, errors="coerce")
                min_val = _safe_scalar(parsed_dt.min())
                max_val = _safe_scalar(parsed_dt.max())
            except Exception:
                pass
        elif ser.dtype == "object":
            try:
                parsed = pd.to_datetime(ser, errors="coerce")
                if parsed.notna().sum() / max(1, ser.notna().sum()) > 0.5:
                    is_dt = True
                    min_val = _safe_scalar(parsed.min())
                    max_val = _safe_scalar(parsed.max())
            except Exception:
                pass

        # Numeric stats (with optional sampling for quantiles)
        if is_num or (ser.dtype == "object" and infer_numeric_from_strings and ser.notna().sum() > 0):
            vals = pd.to_numeric(ser, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if not vals.empty:
                vals_for_stats = vals.sample(n=sample_n, random_state=42) if (sample_n and len(vals) > sample_n) else vals
                try:
                    min_val = float(vals.min());  max_val = float(vals.max())
                    mean = float(vals.mean());    std = float(vals.std())
                    p05 = float(vals_for_stats.quantile(0.05)); p10 = float(vals_for_stats.quantile(0.10))
                    p25 = float(vals_for_stats.quantile(0.25)); p50 = float(vals_for_stats.quantile(0.50)); p75 = float(vals_for_stats.quantile(0.75))
                    p90 = float(vals_for_stats.quantile(0.90)); p95 = float(vals_for_stats.quantile(0.95))

                    iqr = p75 - p25
                    decile_range = p90 - p10

                    # Tukey's fences (k=1.5 "outlier", k=3 "far out")
                    if pd.notna(iqr) and iqr != 0:
                        lb = p25 - 1.5 * iqr; ub = p75 + 1.5 * iqr
                        outlier_lower_count = int((vals < lb).sum())
                        outlier_upper_count = int((vals > ub).sum())
                        outlier_count = outlier_lower_count + outlier_upper_count
                        outlier_ratio = float(outlier_count / max(1, len(vals)))

                        elb = p25 - 3.0 * iqr; eub = p75 + 3.0 * iqr
                        extreme_outlier_count = int(((vals < elb) | (vals > eub)).sum())
                    else:
                        outlier_lower_count = outlier_upper_count = outlier_count = 0
                        outlier_ratio = 0.0; extreme_outlier_count = 0

                    zeros_count = int((vals == 0).sum())
                    neg_count   = int((vals < 0).sum())
                    pos_count   = int((vals > 0).sum())
                    skew = float(vals.skew()) if hasattr(vals, "skew") else np.nan
                    kurt = float(vals.kurt()) if hasattr(vals, "kurt") else np.nan
                except Exception:
                    pass

        # Top/sample values for non-numeric
        try:
            if not is_num and not is_bool:
                try:
                    vc = ser.dropna().value_counts().head(5)
                    top_values = "; ".join([f"{v} ({int(c)})" for v, c in zip(vc.index.astype(str), vc.values)])
                except TypeError:
                    vc = ser.dropna().astype(str).value_counts().head(5)
                    top_values = "; ".join([f"{v} ({int(c)})" for v, c in zip(vc.index, vc.values)])
                sample_values = ", ".join(map(str, ser.dropna().unique()[:5]))
        except Exception:
            top_values = None; sample_values = None

        rows.append({
            "col_index": int(i), "feature": col, "dtype": str(dtype),
            "non_null": int(non_null), "missing": int(missing), "missing_ratio": float(missing_ratio),
            "unique_count": int(unique_count), "is_numeric": bool(is_num), "is_datetime": bool(is_dt), "is_bool": bool(is_bool),
            "top_values": top_values, "sample_values": sample_values,
            "min": _safe_scalar(min_val), "p05": _safe_scalar(p05), "p10": _safe_scalar(p10), "p25": _safe_scalar(p25),
            "p50": _safe_scalar(p50), "p75": _safe_scalar(p75), "p90": _safe_scalar(p90), "p95": _safe_scalar(p95),
            "iqr": _safe_scalar(iqr), "decile_range": _safe_scalar(decile_range),
            "outlier_lower_count": _safe_scalar(outlier_lower_count), "outlier_upper_count": _safe_scalar(outlier_upper_count),
            "outlier_count": _safe_scalar(outlier_count), "outlier_ratio": _safe_scalar(outlier_ratio),
            "extreme_outlier_count": _safe_scalar(extreme_outlier_count),
            "zeros_count": _safe_scalar(zeros_count), "neg_count": _safe_scalar(neg_count), "pos_count": _safe_scalar(pos_count),
            "skew": _safe_scalar(skew), "kurt": _safe_scalar(kurt),
            "true_count": _safe_scalar(true_count), "false_count": _safe_scalar(false_count), "true_ratio": _safe_scalar(true_ratio),
            "memory_bytes": _safe_scalar(memory_bytes), "max": _safe_scalar(max_val), "mean": _safe_scalar(mean), "std": _safe_scalar(std),
        })

    summary = pd.DataFrame(rows)
    order_cols = ["col_index","feature","dtype","is_numeric","is_datetime","is_bool","non_null","missing","missing_ratio","unique_count",
                  "top_values","sample_values","min","p05","p10","p25","p50","p75","p90","p95","iqr","decile_range",
                  "outlier_lower_count","outlier_upper_count","outlier_count","outlier_ratio","extreme_outlier_count",
                  "zeros_count","neg_count","pos_count","skew","kurt","true_count","false_count","true_ratio",
                  "memory_bytes","max","mean","std","feature_category","distribution_type"]
    cols_present = [c for c in order_cols if c in summary.columns]
    return summary[cols_present]

#
#%% [markdown]
# ## 5) Feature categorization (discrete/continuous + distribution types)
#
#%%
def categorize_features(summary: pd.DataFrame, iqr_threshold: float = 1.0, skew_threshold: float = 1.0) -> pd.DataFrame:
    summary = summary.copy()
    summary['feature_category'] = 'unknown'
    summary['distribution_type'] = ''

    # use (non_null + missing) to infer total rows if present
    if 'non_null' in summary.columns and 'missing' in summary.columns:
        total_n = (summary['non_null'] + summary['missing']).max()
    else:
        total_n = np.nan

    for idx, row in summary.iterrows():
        dtype = str(row['dtype'])
        is_num = bool(row.get('is_numeric', False))
        is_bool = bool(row.get('is_bool', False))
        unique_count = int(row.get('unique_count', 0))
        unique_ratio = (unique_count / total_n) if (pd.notna(total_n) and total_n > 0) else 0.0

        if is_bool or 'int' in dtype.lower():
            summary.at[idx, 'feature_category'] = 'discrete'
            summary.at[idx, 'distribution_type'] = 'categorical' if is_bool else 'integer'
        elif dtype == 'object' or (not is_num and unique_ratio < 0.05):
            summary.at[idx, 'feature_category'] = 'discrete'
            summary.at[idx, 'distribution_type'] = 'categorical'
        elif is_num:
            summary.at[idx, 'feature_category'] = 'continuous'
            iqr = row.get('iqr', np.nan); skew = row.get('skew', np.nan)
            if pd.isna(iqr) or pd.isna(skew):
                summary.at[idx, 'distribution_type'] = 'continuous_unknown'; continue
            abs_skew = abs(float(skew)); iqr_val = float(iqr)
            if abs_skew > skew_threshold:
                summary.at[idx, 'distribution_type'] = 'skewed'
            elif iqr_val <= iqr_threshold:
                summary.at[idx, 'distribution_type'] = 'normal_small_spread'
            else:
                summary.at[idx, 'distribution_type'] = 'normal_large_spread'
        else:
            summary.at[idx, 'feature_category'] = 'other'
            summary.at[idx, 'distribution_type'] = 'unknown'
    return summary

def summarize_categorization(summary: pd.DataFrame) -> str:
    lines = []
    lines.append("\n=== Feature Categorization Summary ===\n")
    cat_counts = summary['feature_category'].value_counts()
    lines.append("Feature Categories:")
    for cat, count in cat_counts.items():
        lines.append(f"  {cat}: {count}")

    lines.append("\nDistribution Types (within categories):")
    discrete = summary[summary['feature_category'] == 'discrete']
    if len(discrete) > 0:
        lines.append(f"\n  Discrete features ({len(discrete)}):")
        dist_counts = discrete['distribution_type'].value_counts()
        for dist, count in dist_counts.items():
            lines.append(f"    - {dist}: {count}")
            examples = discrete[discrete['distribution_type'] == dist]['feature'].head(3).tolist()
            lines.append(f"      Examples: {', '.join(map(str, examples))}")

    continuous = summary[summary['feature_category'] == 'continuous']
    if len(continuous) > 0:
        lines.append(f"\n  Continuous features ({len(continuous)}):")
        dist_counts = continuous['distribution_type'].value_counts()
        for dist, count in dist_counts.items():
            lines.append(f"    - {dist}: {count}")
            ex_df = continuous[continuous['distribution_type'] == dist][['feature','iqr','skew']].head(3)
            lines.append("      Examples:")
            for _, ex in ex_df.iterrows():
                iqr_s = f"{ex['iqr']:.3f}" if pd.notna(ex['iqr']) else "NA"
                skew_s = f"{ex['skew']:.3f}" if pd.notna(ex['skew']) else "NA"
                lines.append(f"        {ex['feature']} (IQR={iqr_s}, skew={skew_s})")
    return "\n".join(lines)

#
#%% [markdown]
# ## 6) Plain-text/Markdown report helpers
#
#%%
def _df_to_markdown_table(df: pd.DataFrame, cols=None, max_rows=10) -> str:
    if cols is None:
        cols = df.columns.tolist()
    sub = df[cols] if max_rows is None else df[cols].head(max_rows)
    try:
        return sub.to_markdown(index=False)
    except Exception:
        header = "| " + " | ".join(cols) + " |\n"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
        rows = []
        if max_rows is None:
            sub = df[cols]
        for _, r in sub.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                cells.append("" if pd.isna(v) else str(v))
            rows.append("| " + " | ".join(cells) + " |\n")
        return header + sep + "".join(rows)

def _find_fig(fname_contains: str, figs_dir: Path | None = None) -> Optional[str]:
    figs_dir = Path(figs_dir) if figs_dir is not None else FIGS_DIR
    if not figs_dir.exists():
        return None
    for p in sorted(figs_dir.iterdir()):
        if fname_contains in p.name.lower():
            return p.name
    return None

def write_text_report(summary: pd.DataFrame, out_path: Path, df: Optional[pd.DataFrame] = None, top_n_missing: int = 20):
    lines = []
    lines.append(f"Feature report for {DATA_PATH.name}")
    if df is not None:
        lines.append(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    lines.append("")

    if 'feature_category' in summary.columns:
        lines.append("=== Feature Categorization ===")
        cat_counts = summary['feature_category'].value_counts()
        for cat, count in cat_counts.items():
            lines.append(f"  {cat}: {count}")
        if 'distribution_type' in summary.columns:
            lines.append("\nDistribution Types:")
            for k, v in summary['distribution_type'].value_counts().items():
                lines.append(f"  - {k}: {v}")
        lines.append("")

    lines.append(f"Top {top_n_missing} features by missing ratio:")
    for _, r in summary.sort_values("missing_ratio", ascending=False).head(top_n_missing).iterrows():
        mr = r.get("missing_ratio", None)
        mr_s = "NA" if (mr is None or (isinstance(mr, float) and pd.isna(mr))) else f"{float(mr):.2%}"
        idx = r.get("col_index", None)
        head = f"[{int(idx)}] " if idx is not None else ""
        lines.append(f"- {head}{r['feature']}: missing {int(r['missing'])} ({mr_s}), dtype={r['dtype']}, unique={int(r['unique_count'])}")

    lines.append("")
    lines.append("Per-feature concise summary:")
    for _, r in summary.iterrows():
        feat = r["feature"]; dtype = r["dtype"]
        miss = int(r["missing"]) if not pd.isna(r["missing"]) else "NA"
        miss_ratio = f"{r['missing_ratio']:.2%}" if not pd.isna(r["missing_ratio"]) else "NA"
        if r.get("is_numeric") or r.get("is_datetime"):
            _min = r.get("min",""); _max = r.get("max",""); med = r.get("p50",""); iqr = r.get("iqr",""); dec = r.get("decile_range","")
            out_r = r.get("outlier_ratio", None)
            out_s = "NA" if (out_r is None or (isinstance(out_r,float) and pd.isna(out_r))) else f"{float(out_r):.2%}"
            lines.append(f"{feat} | {dtype} | missing: {miss} ({miss_ratio}) | range: { _min } -> { _max } | median: {med} | IQR: {iqr} | decile_range: {dec} | outlier_ratio: {out_s}")
        elif r.get("is_bool"):
            tr = r.get("true_ratio", None)
            tr_s = "NA" if (tr is None or (isinstance(tr,float) and pd.isna(tr))) else f"{float(tr):.2%}"
            lines.append(f"{feat} | {dtype} | missing: {miss} ({miss_ratio}) | true: {r.get('true_count','NA')} | false: {r.get('false_count','NA')} | true_ratio: {tr_s}")
        else:
            uniq = int(r.get("unique_count") or 0)
            top = r.get("top_values") or ""
            lines.append(f"{feat} | {dtype} | missing: {miss} ({miss_ratio}) | unique: {uniq} | top: {top}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

def generate_markdown_report(summary: pd.DataFrame, out_md: Path | None = None, figs_dir: Path | None = None, data_path: Path | None = None):
    out_md = Path(out_md) if out_md is not None else OUT_MD
    figs_dir = Path(figs_dir) if figs_dir is not None else FIGS_DIR
    data_path = Path(data_path) if data_path is not None else DATA_PATH

    now = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
    lines = []
    lines.append(f"# Feature Report for {data_path.name}\n")
    lines.append(f"_Generated: {now}_\n")

    lines.append("## Dataset summary\n")
    try:
        rows = len(pd.read_csv(data_path))
    except Exception:
        rows = "?"
    lines.append(f"- rows: {rows}\n")
    lines.append(f"- features: {len(summary)}\n")
    if "memory_bytes" in summary.columns:
        try:
            total_mem = int(summary['memory_bytes'].sum(skipna=True))
            lines.append(f"- estimated memory (sum of columns): {total_mem:,} bytes\n")
        except Exception:
            pass

    if 'feature_category' in summary.columns:
        lines.append("\n## Feature Categorization\n")
        cat_counts = summary['feature_category'].value_counts()
        lines.append("| Category | Count |\n| --- | --- |\n")
        for cat, count in cat_counts.items():
            lines.append(f"| {cat} | {count} |\n")

        if 'distribution_type' in summary.columns:
            lines.append("\n### Distribution Types\n")
            dtype_counts = summary['distribution_type'].value_counts()
            lines.append("| Type | Count |\n| --- | --- |\n")
            for dtype, count in dtype_counts.items():
                lines.append(f"| {dtype} | {count} |\n")

    lines.append("\n## Top missing features\n")
    cols = [c for c in ["col_index","feature","missing","missing_ratio","unique_count"] if c in summary.columns]
    lines.append(_df_to_markdown_table(summary.sort_values("missing_ratio", ascending=False), cols=cols, max_rows=20))

    lines.append("\n## Numeric spread and outliers\n")
    try:
        num_table = summary[summary.get('is_numeric')==True].sort_values('iqr', ascending=False)
        num_cols = [c for c in ["col_index","feature","iqr","decile_range","outlier_ratio"] if c in summary.columns]
        lines.append(_df_to_markdown_table(num_table, cols=num_cols, max_rows=20))
    except Exception:
        pass

    lines.append("\n## Per-feature concise summary\n")
    per_cols = [c for c in ["col_index","feature","dtype","feature_category","distribution_type","missing","missing_ratio","unique_count","min","max","p50","iqr","decile_range","outlier_ratio","skew"] if c in summary.columns]
    try:
        lines.append(_df_to_markdown_table(summary[per_cols], cols=per_cols, max_rows=None))
    except Exception:
        lines.append(_df_to_markdown_table(summary, cols=["feature","missing","missing_ratio"], max_rows=None))

    lines.append("\n## Files\n")
    lines.append(f"- Detailed CSV summary: `data_profiling_output/{OUT_SUMMARY_CSV.name}`\n")
    lines.append(f"- Plain text report: `data_profiling_output/{OUT_TEXT.name}`\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md

#
#%% [markdown]
# ## 7) Run profiling + categorization, and write reports (**run this cell**)
#
#%%
section("Describe features")
summary = describe_dataframe(df, sample_n=None, infer_numeric_from_strings=True)

section("Categorize features")
summary = categorize_features(summary, iqr_threshold=1.0, skew_threshold=1.0)

summary_sorted = summary.sort_values("missing_ratio", ascending=False)
show_interactive_summary(summary, df=df)

subtitle("Top 20 columns by missing ratio:")
display(summary_sorted[[c for c in ["feature","dtype","missing","missing_ratio","unique_count"] if c in summary_sorted.columns]].head(20))

summary.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
write_text_report(summary, OUT_TEXT, df)
md_path = generate_markdown_report(summary, out_md=OUT_MD, figs_dir=FIGS_DIR, data_path=DATA_PATH)

print(f"[SAVE] CSV: {OUT_SUMMARY_CSV}")
print(f"[SAVE] TXT: {OUT_TEXT}")
print(f"[SAVE] MD : {md_path}")

#
#%% [markdown]
# ## 8) (Optional) Quick visuals: numeric histograms & top-K bars
# - To avoid too many plots, we only show the first `N_HIST_NUM` numeric columns and the first `N_BAR_CAT` categorical columns.
# - No color specified (keep defaults).
#
#%%
N_HIST_NUM = 6; TOPK = 15; N_BAR_CAT = 6; BINS = 30

# Choose one representative feature per category/distribution and plot it with the
# feature name included in the title. Targets:
# - continuous features grouped by distribution_type
# - a discrete (categorical/integer) feature
# - a boolean feature (if present)
# - a datetime feature (if present)
examples = []
num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if (df[c].dtype == "object") or (str(df[c].dtype) == "category")]

if 'feature_category' in summary.columns:
    # continuous: pick one per distribution type
    cont = summary[summary['feature_category'] == 'continuous']
    if not cont.empty:
        for dist in cont['distribution_type'].dropna().unique():
            sel = cont[cont['distribution_type'] == dist]
            if not sel.empty:
                examples.append(('continuous', dist, sel['feature'].iloc[0]))

    # discrete: pick the first discrete feature
    disc = summary[summary['feature_category'] == 'discrete']
    if not disc.empty:
        examples.append(('discrete', 'categorical', disc['feature'].iloc[0]))

    # boolean
    if 'is_bool' in summary.columns:
        b = summary[summary['is_bool'] == True]
        if not b.empty:
            examples.append(('bool', 'bool', b['feature'].iloc[0]))

    # datetime
    if 'is_datetime' in summary.columns:
        dts = summary[summary['is_datetime'] == True]
        if not dts.empty:
            examples.append(('datetime', 'datetime', dts['feature'].iloc[0]))

# Fallbacks if summary doesn't contain the needed info
if not examples:
    if num_cols:
        examples.append(('continuous', 'unknown', num_cols[0]))
    if cat_cols:
        examples.append(('discrete', 'categorical', cat_cols[0]))

# Plot selected examples (robust to missing/parse errors)
for kind, subtype, col in examples:
    if col not in df.columns:
        continue
    try:
        if kind == 'continuous':
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            if vals.empty:
                continue
            plt.figure()
            vals.hist(bins=BINS)
            plt.title(f"{col} ({subtype})")
            plt.xlabel(col); plt.ylabel('Count')
            plt.tight_layout(); plt.show()

        elif kind in ('discrete', 'bool'):
            vc = df[col].fillna('<NA>').value_counts().head(TOPK)
            plt.figure(figsize=(8,4))
            vc.plot(kind='bar')
            plt.title(f"{col} ({subtype})")
            plt.xlabel(col); plt.ylabel('Count')
            plt.tight_layout(); plt.show()

        elif kind == 'datetime':
            parsed = pd.to_datetime(df[col], errors='coerce').dropna()
            if parsed.empty:
                continue
            # Aggregate by year (or month if more granular range is desired)
            agg = parsed.dt.year.value_counts().sort_index()
            plt.figure(figsize=(8,4))
            agg.plot(kind='bar')
            plt.title(f"{col} (year counts)")
            plt.xlabel('Year'); plt.ylabel('Count')
            plt.tight_layout(); plt.show()

    except Exception:
        # skip on any plotting error to keep the script robust
        continue

# %%

#%% [markdown]
# ## 9) Target analysis (explicit column; simple profile)
# Set your target column name here (no auto-detection / no _looks_like_id)
#%%

TARGET_COL = "target"   # <-- change me

from pandas.api.types import is_numeric_dtype, is_bool_dtype

def profile_target(df: pd.DataFrame, col: str, top_k: int = 20):
    s = df[col]
    n = len(s)
    n_missing = int(s.isna().sum())
    n_unique  = int(s.nunique(dropna=True))
    dtype     = s.dtype

    report = {
        "column": col,
        "dtype": str(dtype),
        "rows": n,
        "missing": n_missing,
        "missing_ratio": float(n_missing / n) if n else float("nan"),
        "unique": n_unique,
    }

    # Simple rule: treat as classification if boolean / non-numeric / small integer cardinality;
    # otherwise treat as regression-like numeric.
    is_classy = (
        is_bool_dtype(s)
        or (not is_numeric_dtype(s))
        or (pd.api.types.is_integer_dtype(s) and n_unique <= 20)
    )

    if is_classy:
        report["kind"] = "classification"
        vc = s.astype("object").value_counts(dropna=True)
        denom = max(1, n - n_missing)
        report["n_classes"] = int(len(vc))
        report["majority_class"] = vc.idxmax() if len(vc) else None
        report["majority_ratio"] = float(vc.max() / denom) if len(vc) else float("nan")
        report["imbalance_ratio_max_min"] = (
            float(vc.max() / vc.min()) if len(vc) >= 2 and vc.min() > 0 else float("nan")
        )
        # Per-class counts/ratios
        report["class_counts"] = {str(k): int(v) for k, v in vc.items()}
        report["class_ratios"] = {str(k): float(v/denom) for k, v in vc.items()}

        # Quick bar plot
        ax = s.fillna("<NA>").astype("object").value_counts().head(top_k).plot(kind="bar")
        ax.set_title(f"Target distribution: {col}")
        ax.set_xlabel(col); ax.set_ylabel("Count")
        plt.tight_layout(); plt.show()

    else:
        report["kind"] = "regression"
        s_num = pd.to_numeric(s, errors="coerce")
        report.update({
            "min": float(s_num.min()),
            "max": float(s_num.max()),
            "mean": float(s_num.mean()),
            "std": float(s_num.std()),
            "p25": float(s_num.quantile(0.25)),
            "p50": float(s_num.quantile(0.50)),
            "p75": float(s_num.quantile(0.75)),
            "skew": float(s_num.skew()),
            "kurt": float(s_num.kurt()),
        })

        # Quick histogram
        s_num.dropna().hist(bins=30)
        plt.title(f"Target histogram: {col}")
        plt.xlabel(col); plt.ylabel("Count")
        plt.tight_layout(); plt.show()

    # Compact table view
    display(pd.DataFrame([report]).T.rename(columns={0: "value"}))
    return report

# Run it:
target_profile = profile_target(df, TARGET_COL)

if TARGET_COL in df.columns:
    s_target = df[TARGET_COL].fillna('<NA>').astype(object)
    vc = s_target.value_counts(dropna=False)
    pct = vc / vc.sum()
    tbl = pd.DataFrame({
        'class': vc.index.astype(str),
        'count': vc.values,
        'pct': (pct.values * 100),
    })
    tbl['cum_pct'] = tbl['pct'].cumsum()

    # Pretty-print for notebook / script
    print('\nTarget class distribution:')
    display(tbl.rename(columns={'pct': 'pct_%', 'cum_pct': 'cum_pct_%'}))

    # Save CSV for downstream use
    out_csv = PROD_DIR / f"target_{TARGET_COL}_class_table.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"[SAVE] Target class table CSV: {out_csv}")

    # Append a markdown table to the main report for quick reader consumption
    try:
        md_lines = []
        md_lines.append('\n## Target: class distribution\n')
        md_lines.append(f'- target column: `{TARGET_COL}`\n')
        md_lines.append(tbl.rename(columns={'pct':'pct_%','cum_pct':'cum_pct_%'}).to_markdown(index=False))
        if OUT_MD.exists():
            OUT_MD.write_text(OUT_MD.read_text(encoding='utf-8') + '\n' + '\n'.join(md_lines), encoding='utf-8')
            print(f"Appended target class table to: {OUT_MD}")
    except Exception as e:
        print('Could not append class table to markdown report:', e)

else:
    print(f"TARGET_COL '{TARGET_COL}' not found in the dataframe columns: {df.columns.tolist()}")

#%%