#!/usr/bin/env python3
"""
test_avatar_on_aids.py

Quick test of the AvatarSynthesizer on the AIDS clinical trial dataset.

Usage:
  ./test_avatar_on_aids.py

Assumes aids_original_data.csv is in the current folder or parent.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

from avatar_synth import AvatarSynthesizer, AvatarSynthConfig, compute_dcr_nndr


def load_aids():
    candidates = [
        "aids_original_data.csv",
        os.path.join("..", "aids_original_data.csv"),
    ]
    csv_path = next((p for p in candidates if os.path.isfile(p)), None)
    if csv_path is None:
        raise FileNotFoundError("Cannot find aids_original_data.csv.")
    df = pd.read_csv(csv_path, sep=";")
    df.columns = df.columns.str.strip().str.lower()
    return df, os.path.abspath(csv_path)


def main():
    df, path = load_aids()
    print(f"Loaded AIDS dataset from {path}, shape={df.shape}")

    # Choose numeric columns, drop pidnum (ID)
    numeric_cols = [
        c for c in df.columns
        if c != "pidnum" and pd.api.types.is_numeric_dtype(df[c])
    ]

    cfg = AvatarSynthConfig(k_neighbors=20, random_state=123)
    synth = AvatarSynthesizer(cfg).fit(df, numeric_cols=numeric_cols, exclude_cols=["pidnum"])
    df_avatar = synth.transform()

    print("\nPreview of original vs avatar (first 3 rows, selected cols):")
    cols_show = ["age", "gender", "race", "cd40", "cd420", "days"]
    print("Original:")
    print(df[cols_show].head(3))
    print("\nAvatar:")
    print(df_avatar[cols_show].head(3))

    # --- Privacy metrics on numeric space ---
    X_orig = df[numeric_cols].to_numpy(dtype=float)
    X_syn = df_avatar[numeric_cols].to_numpy(dtype=float)
    dcr_stats, nndr_stats = compute_dcr_nndr(X_orig, X_syn)

    print("\nDCR (distance to closest original record):")
    print(dcr_stats.to_string(index=False))
    print("\nNNDR (ratio d1/d2; closer to 1 is better privacy):")
    print(nndr_stats.to_string(index=False))

    # --- Utility test: early CD4 response (similar to your Step 6) ---

    df_u = df.copy()
    df_u = df_u[~df_u["cd420"].isna()]
    df_u["y_resp"] = (df_u["cd420"] >= 350).astype(int)

    preds = ["age", "wtkg", "karnof", "preanti", "gender", "race", "drugs", "hemo", "homo", "cd40"]
    preds = [c for c in preds if c in df_u.columns]
    df_u = df_u[["y_resp"] + preds].dropna().copy()

    # Align avatar dataset to the same rows/columns (simplest: regenerate on df_u subset)
    synth_u = AvatarSynthesizer(cfg).fit(df_u, numeric_cols=preds, exclude_cols=[])
    df_u_avatar = synth_u.transform()

    def fit_logit(dfX):
        y = dfX["y_resp"].values
        X = dfX.drop(columns=["y_resp"]).astype(float)
        X = sm.add_constant(X, has_constant="add")
        model = sm.Logit(y, X).fit(disp=False)
        p = model.predict(X)
        return model, p

    def brier_score(y_true, p_hat):
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(p_hat, dtype=float)
        return float(np.mean((p - y) ** 2))

    def auc_rank(y_true, scores):
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

    model_orig, p_orig = fit_logit(df_u)
    model_syn, p_syn = fit_logit(df_u_avatar)

    auc_orig = auc_rank(df_u["y_resp"], p_orig)
    auc_syn = auc_rank(df_u_avatar["y_resp"], p_syn)
    bs_orig = brier_score(df_u["y_resp"], p_orig)
    bs_syn = brier_score(df_u_avatar["y_resp"], p_syn)

    print("\nUtility: predicting early CD4 response (cd420 ≥ 350)")
    print(f"  AUC  original vs avatar: {auc_orig:.3f}  →  {auc_syn:.3f} (higher is better)")
    print(f"  Brier original vs avatar: {bs_orig:.3f}  →  {bs_syn:.3f} (lower is better)")

    # Compare coefficients quickly
    def coef_table(model, label):
        s = pd.Series(model.params, name=f"coef_{label}")
        return s

    ct = pd.concat(
        [coef_table(model_orig, "orig"), coef_table(model_syn, "avatar")],
        axis=1,
    )
    print("\nLogistic regression coefficients (log-odds) original vs avatar:")
    print(ct.round(3))


if __name__ == "__main__":
    main()

