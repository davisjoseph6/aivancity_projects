# Anonymization & Patient-Centric Synthetic Data (Avatar) Project

This project explores, assesses, and anonymizes `aids_original_data.csv`, and implements the **Avatar** patient-centric synthetic data method on the same dataset, following:

> Guillaudeux et al., *Patient-centric synthetic data generation, no reason to risk re-identification in biomedical data analysis*, npj Digital Medicine, 2023.

The repository now contains two complementary tracks:

* **Steps 1–4 (Python + R):** classical anonymization pipeline (exploration, risk analysis with sdcMicro, anonymization and risk–utility trade-offs).
* **Step 6 (Python):** downstream clinical use case demonstration on anonymized data.
* **Avatar track (Python):** patient-centric **synthetic data generation** on the AIDS trial, with privacy and utility evaluation inspired by Guillaudeux et al. (2023).

> 🔎 **Outputs (PNGs/CSVs) are versioned in Git** so collaborators can review them without re-running, *including* the synthetic dataset (`aids_avatar_k20.csv`) and Avatar figures.

---

## 1) Prerequisites

* **Python** 3.10–3.12
* **R** 4.2+ (with internet access to install packages from CRAN)
* Git

### Python packages (from `requirements.txt`)

```bash
numpy>=2.2,<2.3
pandas>=2.2,<2.3
matplotlib>=3.8,<3.11
seaborn>=0.13.2,<0.14
scipy>=1.14.1,<1.15
statsmodels>=0.14.3,<0.15
```

### R packages

* `sdcMicro` (the Step 3 script installs it automatically if missing)

---

## 2) Set up the Python environment

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version        # confirm 3.10–3.12
pip install -U pip
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
py -m venv .venv
. .\.venv\Scripts\Activate.ps1
python --version        # confirm 3.10–3.12
python -m pip install -U pip
pip install -r requirements.txt
```

---

## 3) Data: how the CSV is handled

* **Raw data:** `aids_original_data.csv` (AIDS clinical trial).
* **Synthetic data (Avatar):** `aids_avatar_k20.csv` (safe to commit – fully synthetic).
* **Paper:** `digital_medicine_2023.pdf` (local copy of Guillaudeux et al. 2023 for reference).

For confidentiality reasons, **do not commit** `aids_original_data.csv` to a public repository.

Scripts will typically look for the raw CSV in:

1. `./aids_original_data.csv`
2. `./data/aids_original_data.csv`

Older Step 1–4 scripts may auto-download the public AIDS CSV into `./data/` if not found. New Avatar scripts (`test_avatar_on_aids.py`, `avatar_plots.py`) assume the file is present locally and will raise a `FileNotFoundError` if it is missing.

If you already committed the raw CSV in the past, stop tracking it:

```bash
git rm --cached aids_original_data.csv
git commit -m "Stop tracking raw dataset"
```

The **synthetic** dataset `aids_avatar_k20.csv` and all PNG figures **can and should be versioned** in Git.

---

## 4) Run the pipeline

### 4.1 Step 1–2 (Explore & Visualize) — Python

```bash
cd step1_and_step2
python explore_visualize_aids.py
```

**Outputs (committed):**

* `step1_and_step2/outputs/figs/*.png`
* `step1_and_step2/outputs/csv/*.csv`

---

### 4.2 Step 3 (Risk Analysis) — R

Open **RStudio** (or R console):

```r
setwd("step_3")
source("step3_visual_analysis_EN.R")
```

This will:

* Build an `sdcMicro` object with QIs = age, gender, race.
* Print global risk, expected re-IDs, % unique, % with k ≤ 5.
* Check attribute disclosure (arms → treat).
* Save minimal outputs with plain-language captions.

**Outputs (committed):**

* `outputs/plots/*.png`
* `outputs/csv/*.csv`

If your R working directory is different (e.g., Documents), adjust the script’s output path or move the files into the repo before committing.

---

### 4.3 Step 4 (Anonymization methods) — Python

```bash
cd step4
python step4_anonymization.py
```

**Outputs (committed):**

* `step4/step4_results_summary.csv`
* `step4/step4_plot_risk_by_method.png`
* `step4/step4_plot_utility_by_method.png`
* `step4/step4_plot_risk_utility_tradeoff.png`

---

### 4.4 Step 6 (Use case demonstration) — Python

```bash
cd step_6
python step6_use_case.py
```

**Outputs:**

* PNG: `step6_coefficients_comparison.png`, `step6_roc_curves.png`, `step6_subgroup_rates_by_ageband.png`
* CSV: `step6_model_metrics.csv`, `step6_feature_effects.csv`, `step6_subgroup_summary.csv`

**Interpretation:**

* If AUC/Brier are close between original and anonymized, anonymization preserved predictive utility.
* Similar coefficients → relationships preserved.
* Similar age-band rates → subgroup conclusions preserved.

---

### 4.5 Avatar synthetic data (patient-centric) — Python

This track implements a **numeric-only** version of the Avatar method from Guillaudeux et al. (2023) on the AIDS clinical trial dataset. It:

* Builds a standardized numeric latent space (excluding `pidnum`).
* For each patient, finds `k_neighbors` nearest neighbours.
* Generates a local, stochastic mixture (one synthetic "avatar" per patient).
* Evaluates **privacy** (DCR, NNDR) and **utility** (early CD4 response model).

From the **repo root**:

```bash
# Quick Avatar sanity test + synthetic dataset
./test_avatar_on_aids.py > test_avatar_log.txt

# Full privacy/utility plots and metrics
./avatar_plots.py
```

Key scripts:

* `avatar_synth.py` – core Avatar implementation (`AvatarSynthesizer`, `AvatarSynthConfig`, `compute_dcr_nndr`).
* `test_avatar_on_aids.py` – quick test: prints preview, DCR/NNDR, and logistic regression comparisons.
* `avatar_plots.py` – generates all Avatar-related figures and prints summary metrics.

**Avatar outputs (committed):**

* `aids_avatar_k20.csv`
  → Patient-centric synthetic AIDS dataset (one avatar per original, `k_neighbors = 20`).
* `fig_avatar_marginals_original_vs_synth.png`
  → Overlaid marginal distributions for key variables (age, cd40, cd420, days) in real vs avatar data.
* `fig_avatar_cd40_cd420_scatter.png`
  → Joint distribution of baseline vs week-20 CD4 (cd40 vs cd420) in real vs avatar data.
* `fig_avatar_privacy_dcr_nndr.png`
  → DCR (distance to closest record) and NNDR (nearest-neighbour distance ratio) distributions.
* `fig_avatar_utility_auc_brier.png`
  → AUC and Brier score for early CD4 response model (real vs avatar data).

For a detailed explanation of the method, metrics, and results, see the internal report:

> **“Patient-Centric Synthetic Data for Privacy-Preserving Analysis of the AIDS Clinical Trial: Understanding and Implementing the Avatar Method.”**

This report explains how the Avatar track maps to GDPR/EDPB criteria (singling out, linkability, inference) and describes local cloaking/hidden rate and parameter choices.

---

## 5) Commit & push outputs

After running the scripts, commit the generated outputs so teammates can inspect them without re-running the pipeline.

```bash
git add step1_and_step2/outputs/figs \
        step1_and_step2/outputs/csv \
        outputs/plots \
        outputs/csv \
        step4/*.png \
        step4/*.csv \
        step_6/*.png \
        step_6/*.csv \
        aids_avatar_k20.csv \
        fig_avatar_*.png

git commit -m "Update generated anonymization + Avatar outputs"
git push
```

---

## 6) Reproducibility notes

* PRAM / random mechanisms in anonymization scripts are controlled by a fixed seed inside the scripts.
* Python & R versions are effectively pinned (see `requirements.txt` and CRAN defaults).
* Capture R session info if needed:

```r
sessionInfo()
```

For Avatar:

* `AvatarSynthConfig.random_state` controls randomness in neighbour mixing.
* Re-running with the same seed and `k_neighbors` reproduces the same `aids_avatar_k20.csv` and metrics.

---

## 7) Interpreting key metrics (quick)

For **classical anonymization (Steps 1–4)**:

* **Risk:**

  * % unique
  * % with k ≤ 5
  * Expected re-IDs
    → Lower is safer.
* **Utility:**

  * IL1 (overall change; lower is better)
  * Eigenvalue similarity (structural similarity; higher is better)
* **Best trade-off (in the example scripts):**
  10-year age banding often yields a large risk drop with a small IL1 penalty.

For **Avatar synthetic data**:

* **DCR (Distance to Closest Record):**
  Large distances mean avatars are not tiny perturbations of single real records.
* **NNDR (Nearest-Neighbour Distance Ratio):**
  Values close to 1 mean the closest and second-closest originals are similar in distance → reduces singling out and linkability.
* **Utility (AUC, Brier):**
  If AUC and Brier for the early CD4 response model are close between original and avatar data, the synthetic dataset preserves predictive structure.

---

## 8) Robustness checks you can run quickly

* Different thresholds for response (e.g., `cd420 ≥ 400`) to show conclusions aren’t threshold-sensitive.
* Alternative coarsening (5-year vs 10-year vs 15-year age bands) to illustrate the privacy–utility continuum in this use case.
* Train/test split (or CV) in the use-case scripts so metrics reflect generalization (by default, Step 6 fits and evaluates in-sample to keep dependencies minimal).
* For Avatar, vary `k_neighbors` (e.g., 10, 20, 40) and compare:

  * DCR/NNDR distributions,
  * AUC/Brier,
  * Qualitative similarity of logistic regression coefficients.

---

## 9) Troubleshooting

* **Missing Python packages** → activate `.venv` and `pip install -r requirements.txt`.
* **R can’t find `sdcMicro`** → run `install.packages("sdcMicro", repos = "https://cloud.r-project.org")`.
* **Outputs not appearing** → check terminal `SAVED:` lines for exact paths.
* On Windows, if `statsmodels` errors, ensure you have recent VC++ Build Tools or stick to the pinned versions.
* For Avatar scripts:

  * `FileNotFoundError` for `aids_original_data.csv` → ensure the CSV is in the repo root or adjust the paths.
  * If plots look empty or strange, verify the CSV delimiter (`sep=";"`) and column renaming (`df.columns = df.columns.str.strip().str.lower()`).

---

## 10) Security note

* Do **not** release both `arms` and `treat` together in a way that allows deterministic attribute disclosure.
* Prefer **10-year age bands** in released anonymized datasets; consider **15-year bands** for tighter privacy targets.
* Only share **synthetic** data (`aids_avatar_k20.csv`) and anonymized outputs externally; keep raw `aids_original_data.csv` inside a secure environment.
* Use Avatar privacy metrics (DCR, NNDR and, if implemented, local cloaking/hidden rate) as part of the evidence that the shared dataset no longer allows singling out, linkability, or high-confidence inference about specific individuals.

