#!/usr/bin/env python3
"""
avatar_plots.py

Minimal visualization suite for the Avatar synthetic AIDS dataset.

Generates 4 figures (PNG) in the current directory:

1) fig_avatar_marginals_original_vs_synth.png
   - Overlaid histograms for: age, cd40, cd420, days

2) fig_avatar_cd40_cd420_scatter.png
   - Joint structure of cd40 vs cd420, original vs avatar

3) fig_avatar_privacy_dcr_nndr.png
   - Histograms of Distance to Closest Record (DCR)
     and Nearest Neighbor Distance Ratio (NNDR)

4) fig_avatar_utility_auc_brier.png
   - Bar chart of AUC and Brier score for a simple logistic
     model predicting cd420 >= 350, using original vs avatar data.

Usage:
    (.venv) python3 avatar_plots.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# -----------------------------
# Helpers: load data
# -----------------------------
def load_aids_pair():
    """Load original and avatar AIDS datasets."""
    # Original
    orig_candidates = [
        "aids_original_data.csv",
        os.path.join("..", "aids_original_data.csv"),
    ]
    orig_path = next((p for p in orig_candidates if os.path.isfile(p)), None)
    if orig_path is None:
        raise FileNotFoundError("Cannot find aids_original_data.csv.")
    df_orig = pd.read_csv(orig_path, sep=";")
    df_orig.columns = df_orig.columns.str.strip().str.lower()

    # Avatar (k=20)
    avatar_candidates = [
        "aids_avatar_k20.csv",
        os.path.join("..", "aids_avatar_k20.csv"),
    ]
    avatar_path = next((p for p in avatar_candidates if os.path.isfile(p)), None)
    if avatar_path is None:
        raise FileNotFoundError("Cannot find aids_avatar_k20.csv.")
    df_avatar = pd.read_csv(avatar_path)
    df_avatar.columns = df_avatar.columns.str.strip().str.lower()

    # Ensure same columns / ordering
    common_cols = [c for c in df_orig.columns if c in df_avatar.columns]
    df_orig = df_orig[common_cols]
    df_avatar = df_avatar[common_cols]

    print(f"Loaded original from {os.path.abspath(orig_path)}, shape={df_orig.shape}")
    print(f"Loaded avatar   from {os.path.abspath(avatar_path)}, shape={df_avatar.shape}")
    return df_orig, df_avatar


# -----------------------------
# 1. Marginal distributions
# -----------------------------
def plot_marginals(df_orig, df_avatar):
    """
    Overlaid histograms for a few key variables
    (age, cd40, cd420, days).
    """
    vars_ = ["age", "cd40", "cd420", "days"]
    vars_ = [v for v in vars_ if v in df_orig.columns]

    if not vars_:
        print("No requested variables found for marginals plot; skipping.")
        return

    n = len(vars_)
    nrows = 2
    ncols = 2 if n > 1 else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 6))
    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes, vars_):
        o = df_orig[col].dropna().to_numpy(dtype=float)
        a = df_avatar[col].dropna().to_numpy(dtype=float)
        if len(o) == 0 or len(a) == 0:
            ax.set_title(f"{col} (no data)")
            continue

        data_all = np.concatenate([o, a])
        bins = np.histogram_bin_edges(data_all, bins=30)

        ax.hist(o, bins=bins, alpha=0.5, label="Original", density=True)
        ax.hist(a, bins=bins, alpha=0.5, label="Avatar", density=True)
        ax.set_title(col)
        ax.set_ylabel("Density")

    # Any unused axes
    for j in range(len(vars_), len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Marginal distributions: original vs avatar", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fname = "fig_avatar_marginals_original_vs_synth.png"
    plt.savefig(fname, dpi=300)
    print(f"SAVED: {os.path.abspath(fname)}")


# -----------------------------
# 2. Joint structure (cd40 vs cd420)
# -----------------------------
def plot_cd40_cd420_scatter(df_orig, df_avatar):
    """Scatter plots for cd40 vs cd420 (original vs avatar)."""
    if "cd40" not in df_orig.columns or "cd420" not in df_orig.columns:
        print("cd40/cd420 not both present; skipping joint plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    def scatter_panel(ax, df, title):
        sub = df[["cd40", "cd420"]].dropna()
        ax.scatter(sub["cd40"], sub["cd420"], alpha=0.3, s=10)
        ax.set_title(title)
        ax.set_xlabel("cd40 (baseline)")
        ax.set_ylabel("cd420 (~20-week CD4)")

    scatter_panel(axes[0], df_orig, "Original")
    scatter_panel(axes[1], df_avatar, "Avatar (k=20)")

    fig.suptitle("Joint structure: cd40 vs cd420", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fname = "fig_avatar_cd40_cd420_scatter.png"
    plt.savefig(fname, dpi=300)
    print(f"SAVED: {os.path.abspath(fname)}")


# -----------------------------
# 3. Privacy metrics: DCR & NNDR
# -----------------------------
def compute_dcr_nndr_raw(X_orig, X_syn):
    """
    Compute raw per-row DCR and NNDR arrays, similar to compute_dcr_nndr
    in avatar_synth, but returning the full vectors for plotting.
    """
    X_orig = np.asarray(X_orig, dtype=float)
    X_syn = np.asarray(X_syn, dtype=float)

    # Mean-impute NaNs for distance computations
    if np.isnan(X_orig).any():
        col_means = np.nanmean(X_orig, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        inds = np.where(np.isnan(X_orig))
        X_orig[inds] = col_means[inds[1]]

    if np.isnan(X_syn).any():
        col_means_syn = np.nanmean(X_syn, axis=0)
        col_means_syn = np.nan_to_num(col_means_syn, nan=0.0)
        inds_syn = np.where(np.isnan(X_syn))
        X_syn[inds_syn] = col_means_syn[inds_syn[1]]

    norms_syn = np.sum(X_syn ** 2, axis=1)
    norms_orig = np.sum(X_orig ** 2, axis=1)
    dist2 = norms_syn[:, None] + norms_orig[None, :] - 2 * (X_syn @ X_orig.T)
    dist2 = np.maximum(dist2, 0.0)
    dist = np.sqrt(dist2)

    sorted_d = np.sort(dist, axis=1)
    d1 = sorted_d[:, 0]
    d2 = sorted_d[:, 1]

    dcr = d1
    nndr = d1 / (d2 + 1e-8)
    return dcr, nndr


def plot_privacy_metrics(df_orig, df_avatar):
    """Histograms of DCR and NNDR."""
    # Use all numeric columns except pidnum
    num_cols = [
        c for c in df_orig.columns
        if c != "pidnum" and pd.api.types.is_numeric_dtype(df_orig[c])
    ]
    if not num_cols:
        print("No numeric columns found for privacy metrics; skipping.")
        return

    X_orig = df_orig[num_cols].to_numpy(dtype=float)
    X_syn = df_avatar[num_cols].to_numpy(dtype=float)

    dcr, nndr = compute_dcr_nndr_raw(X_orig, X_syn)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(dcr, bins=30, alpha=0.8)
    axes[0].set_title("DCR (distance to closest original record)")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Count")

    axes[1].hist(nndr, bins=30, alpha=0.8)
    axes[1].set_title("NNDR (d1 / d2)")
    axes[1].set_xlabel("Ratio")
    axes[1].set_ylabel("Count")

    fig.suptitle("Privacy metrics for Avatar synthetic data", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fname = "fig_avatar_privacy_dcr_nndr.png"
    plt.savefig(fname, dpi=300)
    print(f"SAVED: {os.path.abspath(fname)}")

    # Also print basic quantiles for the report text
    def q_stats(x):
        return {
            "median": float(np.median(x)),
            "q05": float(np.quantile(x, 0.05)),
            "q95": float(np.quantile(x, 0.95)),
            "min": float(x.min()),
            "max": float(x.max()),
        }

    print("\nDCR summary:", q_stats(dcr))
    print("NNDR summary:", q_stats(nndr))


# -----------------------------
# 4. Utility: AUC & Brier
# -----------------------------
def auc_rank(y_true, scores):
    """AUC via rank statistic (no sklearn)."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    sum_ranks_pos = ranks[y == 1].sum()
    auc = (sum_ranks_pos - n1 * (n1 + 1) / 2.0) / (n0 * n1)
    return float(auc)


def brier_score(y_true, p_hat):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_hat, dtype=float)
    return float(np.mean((p - y) ** 2))


def fit_logit(dfX):
    y = dfX["y_resp"].values
    X = dfX.drop(columns=["y_resp"]).astype(float)
    X = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y, X).fit(disp=False)
    p = model.predict(X)
    return model, p


def plot_utility_auc_brier(df_orig, df_avatar):
    """
    Fit the same early-CD4-response model on original and avatar data,
    and plot AUC + Brier.
    """
    df = df_orig.copy()
    df = df[~df["cd420"].isna()]
    df["y_resp"] = (df["cd420"] >= 350).astype(int)

    preds = ["age", "wtkg", "karnof", "preanti", "gender", "race", "drugs", "hemo", "homo", "cd40"]
    preds = [c for c in preds if c in df.columns]
    df_u = df[["y_resp"] + preds].dropna().copy()

    # Align avatar data to same feature set
    df_a = df_avatar.copy()
    df_a = df_a[df_a.index.isin(df_u.index)]
    df_a = df_a[preds].copy()
    df_a.insert(0, "y_resp", df_u["y_resp"].values)

    model_orig, p_orig = fit_logit(df_u)
    model_ava, p_ava = fit_logit(df_a)

    auc_o = auc_rank(df_u["y_resp"], p_orig)
    auc_a = auc_rank(df_a["y_resp"], p_ava)
    brier_o = brier_score(df_u["y_resp"], p_orig)
    brier_a = brier_score(df_a["y_resp"], p_ava)

    print("\nUtility metrics (cd420 ≥ 350):")
    print(f"  AUC   original vs avatar: {auc_o:.3f}  →  {auc_a:.3f}")
    print(f"  Brier original vs avatar: {brier_o:.3f}  →  {brier_a:.3f}")

    # Plot as bar chart
    fig, ax = plt.subplots(figsize=(6, 4))

    metrics = ["AUC", "Brier"]
    original_vals = [auc_o, brier_o]
    avatar_vals = [auc_a, brier_a]

    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w / 2.0, original_vals, width=w, label="Original")
    ax.bar(x + w / 2.0, avatar_vals, width=w, label="Avatar")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Model performance: original vs avatar")
    ax.legend()
    fig.tight_layout()

    fname = "fig_avatar_utility_auc_brier.png"
    plt.savefig(fname, dpi=300)
    print(f"SAVED: {os.path.abspath(fname)}")


# -----------------------------
# Main
# -----------------------------
def main():
    df_orig, df_avatar = load_aids_pair()

    plot_marginals(df_orig, df_avatar)
    plot_cd40_cd420_scatter(df_orig, df_avatar)
    plot_privacy_metrics(df_orig, df_avatar)
    plot_utility_auc_brier(df_orig, df_avatar)

    print("\nDone. Generated 4 figures for the Avatar report.")


if __name__ == "__main__":
    main()

