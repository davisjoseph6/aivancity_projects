#!/usr/bin/env python3
"""
Step 4 — Apply anonymization methods, vary parameters, and measure
Disclosure Risk + Information Loss. Saves concise PNGs/CSVs in the
CURRENT directory and prints clear explanations to the terminal.

Methods & parameters:
  • Age banding (global recoding): [5, 10, 15] years
  • PRAM (randomized response) on gender: flip p in [0.01, 0.05, 0.10]
  • PRAM on race: flip p in [0.01, 0.05, 0.10]

Metrics:
  Disclosure risk (QIs = age, gender, race):
    - percent_unique
    - percent_k_le_5
    - avg_linkage_risk_percent (mean(1/k)*100)
    - expected_reids (sum(1/k))
  Utility / information loss:
    - IL1_overall (with IL1_numeric, IL1_categorical)
    - eigenvalue_similarity_percent (correlation-matrix eigenvalues)

Outputs (in current folder):
  - step4_results_summary.csv
  - step4_plot_risk_by_method.png
  - step4_plot_utility_by_method.png
  - step4_plot_risk_utility_tradeoff.png
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Load data (semicolon CSV)
# ---------------------------
CANDIDATES = [
    "aids_original_data.csv",
    os.path.join("..", "aids_original_data.csv"),
]
URL_FALLBACK = "https://raw.githubusercontent.com/octopize/avatar-paper/main/datasets/AIDS/aids_original_data.csv"

csv_path = None
for p in CANDIDATES:
    if os.path.isfile(p):
        csv_path = p
        break

if csv_path is None:
    # Download into current folder to keep everything local
    import urllib.request
    csv_path = "aids_original_data.csv"
    print("CSV not found locally; downloading from GitHub raw ...")
    urllib.request.urlretrieve(URL_FALLBACK, csv_path)

df_orig = pd.read_csv(csv_path, sep=";")
df_orig.columns = df_orig.columns.str.strip().str.lower()

print(f"\nLoaded: {os.path.abspath(csv_path)}")
print("Shape:", df_orig.shape)
print("First 5 rows:\n", df_orig.head(5), "\n")

# ---------------------------
# Helpers
# ---------------------------
rng = np.random.default_rng(12345)

def pram_flip_binary(series, p):
    """Flip a 0/1 binary series with probability p; NaNs preserved."""
    s = series.copy()
    mask = s.notna()
    flips = rng.random(mask.sum()) < p
    s.loc[mask] = (s.loc[mask].astype(int) ^ flips.astype(int))
    return s

def age_band(series, width):
    """Global recode: map age to bin midpoint (e.g., 0-4 -> 2, 5-9 -> 7 for width=5)."""
    s = series.copy()
    mask = s.notna()
    mid = (np.floor(s.loc[mask] / width) * width + (width / 2.0)).astype(int)
    s.loc[mask] = mid
    return s

def k_equivalence_sizes(df_keys):
    """
    For given key columns (no NaNs), return vector of k per row.
    """
    # If NaNs, drop them for risk computation
    keys = df_keys.dropna()
    if keys.empty:
        return pd.Series([], dtype=float)
    counts = keys.value_counts(dropna=False)
    # Map back to the original index positions present in keys
    k_vec = keys.apply(lambda row: counts[tuple(row)], axis=1)
    return k_vec

def disclosure_risk_metrics(df, key_vars):
    """
    Compute: percent_unique, percent_k_le_5, avg_linkage_risk_percent, expected_reids
    on rows where all key_vars are non-missing.
    """
    sub = df[key_vars].dropna()
    if sub.empty:
        return dict(percent_unique=np.nan, percent_k_le_5=np.nan,
                    avg_linkage_risk_percent=np.nan, expected_reids=np.nan,
                    n_keys_evaluated=0)
    k = k_equivalence_sizes(sub)
    percent_unique = (k.eq(1).mean() * 100.0)
    percent_k_le_5 = (k.le(5).mean() * 100.0)
    avg_linkage_risk_percent = (np.mean(1.0 / k) * 100.0)
    expected_reids = float(np.sum(1.0 / k))
    return dict(percent_unique=percent_unique,
                percent_k_le_5=percent_k_le_5,
                avg_linkage_risk_percent=avg_linkage_risk_percent,
                expected_reids=expected_reids,
                n_keys_evaluated=len(k))

def il1_numeric(orig, anon, cols):
    """Mean over numeric columns of mean(|Δ| / (range_orig)) ignoring NaNs."""
    vals = []
    for c in cols:
        if c not in orig.columns or c not in anon.columns:
            continue
        a = orig[c].astype(float)
        b = anon[c].astype(float)
        rng_ = a.max() - a.min()
        if not np.isfinite(rng_) or rng_ == 0:
            continue
        mask = a.notna() & b.notna()
        if mask.any():
            v = np.mean(np.abs(b[mask].values - a[mask].values) / rng_)
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan

def il1_categorical(orig, anon, cols):
    """Mean over categorical columns of proportion changed (orig != anon), ignoring NaNs."""
    vals = []
    for c in cols:
        if c not in orig.columns or c not in anon.columns:
            continue
        a = orig[c]
        b = anon[c]
        mask = a.notna() & b.notna()
        if mask.any():
            v = np.mean((a[mask].astype(str).values != b[mask].astype(str).values).astype(float))
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan

def eigen_similarity_percent(orig, anon, numeric_cols):
    """
    Compare eigenvalues of correlation matrices.
    Return similarity (%) = 100 * (1 - L1(λ' - λ) / sum(λ)),
    where sum(λ) = p for correlation matrix of dimension p.
    """
    cols = [c for c in numeric_cols if c in orig.columns and c in anon.columns]
    if len(cols) < 2:
        return np.nan
    A = orig[cols].corr(method='pearson')
    B = anon[cols].corr(method='pearson')
    if A.isna().values.any() or B.isna().values.any():
        # pairwise deletion fallback
        A = orig[cols].dropna().corr()
        B = anon[cols].dropna().corr()
    try:
        evals_A = np.sort(np.linalg.eigvalsh(A.values))[::-1]
        evals_B = np.sort(np.linalg.eigvalsh(B.values))[::-1]
        denom = np.sum(evals_A)
        if denom <= 0:
            return np.nan
        sim = 100.0 * (1.0 - np.sum(np.abs(evals_B - evals_A)) / denom)
        return float(sim)
    except np.linalg.LinAlgError:
        return np.nan

def savefig(path, caption=None):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"SAVED: {os.path.abspath(path)}")
    if caption:
        print(f"  ↳ Caption: {caption}")

# ---------------------------
# Config: methods & grids
# ---------------------------
KEY_VARS = ["age", "gender", "race"]
NUMERIC_FOR_EIGEN = [c for c in [
    "age","wtkg","preanti","cd40","cd420","cd496","cd80","cd820","days","karnof"
] if c in df_orig.columns]

METHODS = {
    "AGE_BANDING": {"param_name": "width_years", "grid": [5, 10, 15]},
    "PRAM_GENDER": {"param_name": "flip_prob",   "grid": [0.01, 0.05, 0.10]},
    "PRAM_RACE":   {"param_name": "flip_prob",   "grid": [0.01, 0.05, 0.10]},
}

# ---------------------------
# Baseline metrics
# ---------------------------
print("Computing baseline (no anonymization) metrics on QIs (age, gender, race) ...")
baseline_risk = disclosure_risk_metrics(df_orig, KEY_VARS)
baseline_il1_num = 0.0
baseline_il1_cat = 0.0
baseline_il1_all = 0.0
baseline_eigen = eigen_similarity_percent(df_orig, df_orig, NUMERIC_FOR_EIGEN)

summary_rows = []
summary_rows.append({
    "method": "BASELINE",
    "param_name": "",
    "param_value": "",
    **baseline_risk,
    "IL1_numeric": baseline_il1_num,
    "IL1_categorical": baseline_il1_cat,
    "IL1_overall": baseline_il1_all,
    "eigenvalue_similarity_percent": baseline_eigen
})

print("Baseline risk (on QIs):")
for k, v in baseline_risk.items():
    if k != "n_keys_evaluated":
        print(f"  {k}: {v:.2f}" if isinstance(v, (int, float)) else f"  {k}: {v}")
print("")

# ---------------------------
# Run experiments
# ---------------------------
print("Running anonymization experiments and measuring risk/utility ...")

for method, spec in METHODS.items():
    pname = spec["param_name"]
    grid = spec["grid"]
    for val in grid:
        df_anon = df_orig.copy()
        changed_numeric = []
        changed_cats = []

        if method == "AGE_BANDING":
            if "age" in df_anon.columns:
                df_anon["age"] = age_band(df_anon["age"], width=val)
                changed_numeric.append("age")

        elif method == "PRAM_GENDER":
            if "gender" in df_anon.columns:
                df_anon["gender"] = pram_flip_binary(df_anon["gender"], p=val)
                changed_cats.append("gender")

        elif method == "PRAM_RACE":
            if "race" in df_anon.columns:
                # race is coded 0/1 in this dataset; flip with prob p
                df_anon["race"] = pram_flip_binary(df_anon["race"], p=val)
                changed_cats.append("race")

        # Risk on QIs
        risk = disclosure_risk_metrics(df_anon, KEY_VARS)

        # Utility / IL
        il_num = il1_numeric(df_orig, df_anon, changed_numeric) if changed_numeric else np.nan
        il_cat = il1_categorical(df_orig, df_anon, changed_cats) if changed_cats else np.nan
        # Combine IL1 per-variable contributions that exist
        il_vals = [v for v in [il_num, il_cat] if not (v is None or (isinstance(v, float) and np.isnan(v)))]
        il_all = float(np.mean(il_vals)) if il_vals else 0.0

        eigen_sim = eigen_similarity_percent(df_orig, df_anon, NUMERIC_FOR_EIGEN)

        row = {
            "method": method,
            "param_name": pname,
            "param_value": val,
            **risk,
            "IL1_numeric": il_num if not (isinstance(il_num, float) and np.isnan(il_num)) else 0.0,
            "IL1_categorical": il_cat if not (isinstance(il_cat, float) and np.isnan(il_cat)) else 0.0,
            "IL1_overall": il_all,
            "eigenvalue_similarity_percent": eigen_sim
        }
        summary_rows.append(row)

# ---------------------------
# Summarize & save CSV
# ---------------------------
summary = pd.DataFrame(summary_rows)
summary.to_csv("step4_results_summary.csv", index=False)
print("\nSAVED: ", os.path.abspath("step4_results_summary.csv"))
print("\nSummary (first rows):")
print(summary.head(12).to_string(index=False))

# ---------------------------
# Plots (minimal but informative)
# ---------------------------
# Helper: pivot per method
def subset_method(m):
    return summary[summary["method"] == m].copy()

# 1) RISK vs parameter by method (one figure, 3 subplots)
fig = plt.figure(figsize=(12, 4.5))
methods_order = ["AGE_BANDING", "PRAM_GENDER", "PRAM_RACE"]
metrics = [("percent_unique", "% unique (lower is safer)"),
           ("percent_k_le_5", "% with k ≤ 5 (lower is safer)")]

for i, m in enumerate(methods_order, start=1):
    ax = plt.subplot(1, 3, i)
    dfm = subset_method(m)
    x = dfm["param_value"].astype(float).values
    for met, label in metrics:
        y = dfm[met].values
        ax.plot(x, y, marker="o", label=label)
    ax.set_title(m.replace("_", " ").title())
    ax.set_xlabel(dfm["param_name"].iloc[0] if len(dfm) else "parameter")
    ax.set_ylabel("Percentage")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)

caption = ("Each line shows how **disclosure risk** changes as we vary an anonymization setting. "
           "Lower values are safer: fewer unique profiles and fewer small groups (k ≤ 5).")
savefig("step4_plot_risk_by_method.png", caption)

# 2) UTILITY vs parameter by method (IL1 and Eigenvalue similarity)
fig = plt.figure(figsize=(12, 4.5))
for i, m in enumerate(methods_order, start=1):
    ax = plt.subplot(1, 3, i)
    dfm = subset_method(m)
    x = dfm["param_value"].astype(float).values
    y1 = dfm["IL1_overall"].values
    y2 = dfm["eigenvalue_similarity_percent"].values
    ax.plot(x, y1, marker="o", label="IL1 overall (higher = more change)")
    ax.plot(x, y2, marker="s", label="Eigenvalue similarity % (higher = better)")
    ax.set_title(m.replace("_", " ").title())
    ax.set_xlabel(dfm["param_name"].iloc[0] if len(dfm) else "parameter")
    ax.set_ylabel("Score / Percent")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)

caption = ("**IL1** captures how much values changed (normalized). "
           "Lower IL1 means better data fidelity. "
           "**Eigenvalue similarity** compares correlation structure; higher is better.")
savefig("step4_plot_utility_by_method.png", caption)

# 3) TRADE-OFF: expected re-IDs vs IL1 (all methods/params)
fig = plt.figure(figsize=(6.5, 5))
for m, marker in zip(methods_order, ["o", "s", "^"]):
    dfm = subset_method(m)
    plt.scatter(dfm["IL1_overall"], dfm["expected_reids"], marker=marker, label=m.replace("_"," ").title())
plt.xlabel("IL1 overall (higher = more change / less utility)")
plt.ylabel("Expected re-identifications (lower = safer)")
plt.title("Risk–Utility Trade-off across methods")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=9)
caption = ("Points further **left** and **lower** are best: low information loss and low disclosure risk. "
           "Markers show different methods; each point is a parameter setting.")
savefig("step4_plot_risk_utility_tradeoff.png", caption)

print("\nDone. Files created in the current folder:")
print("  - step4_results_summary.csv")
print("  - step4_plot_risk_by_method.png")
print("  - step4_plot_utility_by_method.png")
print("  - step4_plot_risk_utility_tradeoff.png")

# Additional plain-language guidance
print("\nHow to read the outputs (plain language):")
print("• % unique / % with k≤5: smaller is safer (fewer unique or tiny-profile groups).")
print("• Expected re-identifications: smaller is safer (fewer people can be singled out).")
print("• IL1: measures how much values changed overall; smaller = better data quality.")
print("• Eigenvalue similarity: how similar the relationships between numbers remain; higher = better.")
print("• Trade-off plot: pick parameter settings that keep IL1 low while reducing risk.")

