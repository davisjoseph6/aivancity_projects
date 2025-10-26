# Anonymization Project (Steps 1–4)

This project explores, assesses, and anonymizes `aids_original_data.csv`.

- **Step 1–2 (Python):** Explore variables + visualize distributions/relationships
- **Step 3 (R):** Compute disclosure risk (k-anonymity profile, expected re-IDs) and identify risk drivers
- **Step 4 (Python):** Apply anonymization methods, vary parameters, and measure risk–utility trade-offs

> 🔎 **Outputs (PNGs/CSVs) are versioned** in Git so collaborators can review them without re-running.

---

## 1) Prerequisites

- **Python** 3.10–3.12
- **R** 4.2+ (with internet access to install packages from CRAN)
- Git

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
- `sdcMicro` (the Step 3 script installs it automatically if missing)

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


## 3) Data: how the CSV is handled
Do not commit `aids_original_data.csv`. Scripts will try:

1. `./aids_original_data.csv`

2. `./data/aids_original_data.csv`

If not found, they auto-download from the public GitHub URL into `./data/.`

If you already committed the CSV in the past, stop tracking it:

```bash
git rm --cached aids_original_data.csv
git commit -m "Stop tracking raw dataset"
```

## 4) Run the pipeline

### Step 1–2 (Explore & Visualize) — Python
```
cd step1_and_step2
python explore_visualize_aids.py
```

#### Outputs (committed)

- `step1_and_step2/outputs/figs/*.png`

- `step1_and_step2/outputs/csv/*.csv`

### Step 3 (Risk Analysis) — R
Open **RStudio** (or R console):

```r
setwd("step_3")
source("step3_visual_analysis_EN.R")
```

This will:

- Build an sdcMicro object with QIs = age, gender, race

- Print global risk, expected re-IDs, % unique, % with k ≤ 5

- Check attribute disclosure (arms → treat)

- Save minimal outputs with plain-language captions:

#### Outputs (committed)

- `outputs/plots/*.png`

- `outputs/csv/*.csv`

If your R working directory is different (e.g., Documents), adjust the script’s output path or move the files into the repo before committing.

### Step 4 (Anonymization methods) — Python
```bash
cd step4
python step4_anonymization.py
```

#### Outputs (committed)

- `step4/step4_results_summary.csv`

- `step4/step4_plot_risk_by_method.png`

- `step4/step4_plot_utility_by_method.png`

- `step4/step4_plot_risk_utility_tradeoff.png`

### 5) Commit & push outputs
```bash
git add step1_and_step2/outputs/figs \
        step1_and_step2/outputs/csv \
        outputs/plots \
        outputs/csv \
        step4/*.png \
        step4/*.csv
git commit -m "Update generated outputs"
git push
```

### 6) Reproducibility notes

- PRAM randomness is controlled by a fixed seed inside the scripts.

- Python & R versions pinned (see requirements.txt and CRAN defaults).

- Capture R session info if needed:

```r
sessionInfo()
```

### 7) Interpreting key metrics (quick)
- Risk: % unique, % with k ≤ 5, expected re-IDs (lower is safer).

- Utility: IL1 (overall change; lower is better), Eigenvalue similarity (structural similarity; higher is better).

- Best trade-off found: 10-year age banding (large risk drop, small IL1).

### 8) Troubleshooting
- Missing Python packages → activate .venv and pip install -r requirements.txt.

- R can’t find sdcMicro → run install.packages("sdcMicro", repos = "https://cloud.r-project.org").

- Outputs not appearing → check terminal “SAVED:” lines for exact paths.

* On Windows, if statsmodels errors, ensure you have recent VC++ Build Tools or stick to the pinned versions.

### 9) Security note
- Do not release both arms and treat together (deterministic attribute disclosure).

- Prefer 10-year age bands; consider 15-year bands for tighter privacy targets.
