#!/usr/bin/env python3
"""
Step 6 — Use case demonstration:
Predict early immunologic response (cd420 >= 350) with and without anonymization,
and show how anonymization (10-year age banding; avoid arms↔treat leak) affects
predictive utility and subgroup patterns.

Outputs (current folder):
  PNG
    - step6_coefficients_comparison.png
    - step6_roc_curves.png
    - step6_subgroup_rates_by_ageband.png
  CSV
    - step6_model_metrics.csv
    - step6_feature_effects.csv
    - step6_subgroup_summary.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ---------- Load ----------
CANDIDATES = ["aids_original_data.csv", os.path.join("..","aids_original_data.csv"), os.path.join("..","..","aids_original_data.csv")]
csv_path = next((p for p in CANDIDATES if os.path.isfile(p)), None)
if csv_path is None:
    raise FileNotFoundError("Cannot find aids_original_data.csv in current or parent folders.")
df = pd.read_csv(csv_path, sep=";")
df.columns = df.columns.str.strip().str.lower()

print(f"\nLoaded: {os.path.abspath(csv_path)}  shape={df.shape}")
print("This demo uses label: cd420 >= 350 (drop missing cd420).")

# ---------- Define outcome & predictors ----------
# Outcome: immunologic response at ~20 weeks (binary)
df = df.copy()
df = df[~df["cd420"].isna()]  # keep rows with cd420
df["y_resp"] = (df["cd420"] >= 350).astype(int)

# Core predictors (all numerical/binary; avoid arms↔treat leakage by dropping 'arms')
base_predictors = ["age","wtkg","karnof","preanti","gender","race","drugs","hemo","homo","cd40"]
have = [c for c in base_predictors if c in df.columns]
df_model = df[["y_resp"] + have].dropna().copy()

print(f"Using predictors: {have}")
print(f"Usable rows after dropping NA: {df_model.shape[0]}")

# ---------- Build anonymized variant (Step 4 best: 10-year age band) ----------
def age_band(s, width=10):
    return (np.floor(s/width)*width + width/2).astype(int)

df_anon = df_model.copy()
df_anon["age"] = age_band(df_anon["age"], 10)  # coarsen age to 10-year midpoints

# ---------- Utility metrics ----------
def brier_score(y_true, p_hat):
    return float(np.mean((p_hat - y_true) ** 2))

def auc_from_scores(y_true, scores):
    """
    AUC via rank statistic (equivalent to Mann–Whitney U).
    No sklearn dependency.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    # ranks of scores (average ties)
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s)+1)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    sum_ranks_pos = ranks[y == 1].sum()
    auc = (sum_ranks_pos - n1*(n1+1)/2) / (n0*n1)
    return float(auc)

# ---------- Fit logistic models ----------
def fit_logit(dfX):
    y = dfX["y_resp"].values
    X = dfX.drop(columns=["y_resp"]).astype(float)
    X = sm.add_constant(X, has_constant='add')
    model = sm.Logit(y, X).fit(disp=False)
    p = model.predict(X)
    return model, p

model_orig, p_orig = fit_logit(df_model)
model_anon, p_anon = fit_logit(df_anon)

auc_orig = auc_from_scores(df_model["y_resp"].values, p_orig)
auc_anon = auc_from_scores(df_anon["y_resp"].values, p_anon)
bs_orig  = brier_score(df_model["y_resp"].values, p_orig)
bs_anon  = brier_score(df_anon["y_resp"].values, p_anon)

print("\nModel utility (original vs anonymized):")
print(f"  AUC:   {auc_orig:.3f}  →  {auc_anon:.3f} (higher is better)")
print(f"  Brier: {bs_orig:.3f}  →  {bs_anon:.3f} (lower is better)")

# ---------- Save metrics CSV ----------
metrics = pd.DataFrame({
    "metric": ["AUC","Brier"],
    "original": [auc_orig, bs_orig],
    "anonymized_age10": [auc_anon, bs_anon]
})
metrics.to_csv("step6_model_metrics.csv", index=False)
print(f"SAVED CSV: {os.path.abspath('step6_model_metrics.csv')}")

# ---------- Coefficient comparison ----------
def coef_table(model, label):
    params = model.params.copy()
    se = model.bse.copy()
    out = pd.DataFrame({"feature": params.index, f"coef_{label}": params.values, f"se_{label}": se.values})
    return out

ct_orig = coef_table(model_orig, "orig")
ct_anon = coef_table(model_anon, "anon")
coef_join = pd.merge(ct_orig, ct_anon, on="feature", how="outer").fillna(0.0)
# drop constant from plotting, but keep in CSV
coef_join.to_csv("step6_feature_effects.csv", index=False)
print(f"SAVED CSV: {os.path.abspath('step6_feature_effects.csv')}")

plot_df = coef_join[coef_join["feature"] != "const"].copy()
plot_df = plot_df.set_index("feature").sort_index()

plt.figure(figsize=(8.5, 5.5))
x = np.arange(len(plot_df))
w = 0.38
plt.bar(x - w/2, plot_df["coef_orig"], width=w, label="Original")
plt.bar(x + w/2, plot_df["coef_anon"], width=w, label="Anonymized (age 10-yr bands)")
plt.xticks(x, plot_df.index, rotation=30, ha="right")
plt.axhline(0, color="k", linewidth=0.8)
plt.title("Model coefficients: original vs anonymized")
plt.ylabel("Log-odds coefficient")
plt.legend()
plt.tight_layout()
plt.savefig("step6_coefficients_comparison.png", dpi=300)
print(f"SAVED PNG: {os.path.abspath('step6_coefficients_comparison.png')}")
print("  ↳ Caption: Bars compare how each feature influences the odds of response.\n"
      "     Similar heights/directions mean anonymization preserved relationships.")

# ---------- ROC curves (simple) ----------
def roc_curve_simple(y, scores):
    # thresholds from unique scores
    thr = np.unique(scores)
    thr = np.r_[np.inf, thr[::-1], -np.inf]
    tpr, fpr = [], []
    y = np.asarray(y).astype(int)
    P = (y == 1).sum()
    N = (y == 0).sum()
    for t in thr:
        yhat = (scores >= t).astype(int)
        tp = ((yhat == 1) & (y == 1)).sum()
        fp = ((yhat == 1) & (y == 0)).sum()
        tpr.append(tp / P if P else np.nan)
        fpr.append(fp / N if N else np.nan)
    return np.array(fpr), np.array(tpr)

fpr_o, tpr_o = roc_curve_simple(df_model["y_resp"].values, p_orig)
fpr_a, tpr_a = roc_curve_simple(df_anon["y_resp"].values, p_anon)

plt.figure(figsize=(6.5, 5.5))
plt.plot(fpr_o, tpr_o, label=f"Original (AUC={auc_orig:.3f})")
plt.plot(fpr_a, tpr_a, label=f"Anonymized (AUC={auc_anon:.3f})")
plt.plot([0,1],[0,1],"k--", alpha=0.5)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves — original vs anonymized")
plt.legend()
plt.tight_layout()
plt.savefig("step6_roc_curves.png", dpi=300)
print(f"SAVED PNG: {os.path.abspath('step6_roc_curves.png')}")
print("  ↳ Caption: Lines show the trade-off between sensitivity and false alarms.\n"
      "     Close curves mean anonymization had little effect on predictive power.")

# ---------- Subgroup rates by age band ----------
def to_ageband(s, width=10):
    return (np.floor(s/width)*width).astype(int)  # left edge of band (e.g., 30, 40, ...)
df_subg_o = df_model.copy()
df_subg_o["age_band_10y"] = to_ageband(df_subg_o["age"], 10)
df_subg_a = df_anon.copy()
df_subg_a["age_band_10y"] = to_ageband(df_subg_a["age"], 10)  # already banded, consistent

grp_o = df_subg_o.groupby("age_band_10y")["y_resp"].agg(["count","mean"]).rename(columns={"count":"n_orig","mean":"rate_orig"})
grp_a = df_subg_a.groupby("age_band_10y")["y_resp"].agg(["count","mean"]).rename(columns={"count":"n_anon","mean":"rate_anon"})
subg = pd.concat([grp_o, grp_a], axis=1)
subg["delta_pct_points"] = 100*(subg["rate_anon"] - subg["rate_orig"])
subg = subg.reset_index().sort_values("age_band_10y")

subg.to_csv("step6_subgroup_summary.csv", index=False)
print(f"SAVED CSV: {os.path.abspath('step6_subgroup_summary.csv')}")

plt.figure(figsize=(8.5, 5.0))
bands = subg["age_band_10y"].astype(int).astype(str)
x = np.arange(len(subg))
w = 0.38
plt.bar(x - w/2, 100*subg["rate_orig"].values, width=w, label="Original")
plt.bar(x + w/2, 100*subg["rate_anon"].values, width=w, label="Anonymized")
plt.xticks(x, [f"{b}-{int(b)+9}" for b in bands], rotation=0)
plt.ylabel("Response rate (%)")
plt.title("Immunologic response by 10-year age bands")
plt.legend()
plt.tight_layout()
plt.savefig("step6_subgroup_rates_by_ageband.png", dpi=300)
print(f"SAVED PNG: {os.path.abspath('step6_subgroup_rates_by_ageband.png')}")
print("  ↳ Caption: Bars show response rates within 10-yr age groups.\n"
      "     Similar bars indicate subgroup patterns survived anonymization.")

print("\nDONE. Created 3 PNGs and 3 CSVs in the current folder.")
print("How to interpret:")
print(" • If AUC/Brier are close, anonymization preserved predictive utility.")
print(" • If coefficients' signs/magnitude are similar, relationships are preserved.")
print(" • If subgroup bars are close, age-wise conclusions are stable.\n")

