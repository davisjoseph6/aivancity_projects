#!/usr/bin/env Rscript
# ================================================================
# Step 3 — Re-identification Risk Analysis (EN, enhanced + install fix)
#
# What this script does:
# 1) Uses a local, writable R library (./r_libs) so installs work without sudo.
# 2) Loads AIDS CSV (robust path fallback).
# 3) Computes global & individual disclosure risk with sdcMicro.
# 4) Identifies parameters driving risk:
#    - Small-k equivalence classes (k-anonymity)
#    - Rare levels in categorical QIs
#    - Outliers in numeric QIs (|z|>3)
#    - l-diversity and t-closeness (TV distance) for 'treat'
#    - Sensitivity: see how adding {homo|hemo|drugs} to QIs changes small-k shares
# 5) Displays minimal plots (boxplot/histograms) — display-only, no images written.
# 6) Writes a few concise CSVs to outputs_step3/ (set CSV_OUTDIR <- NULL to disable).
#
# Usage:
#   Rscript step3_visual_analysis_EN.R
# ================================================================

# --------------------------
# 0) Use a local, writable package library
# --------------------------
project_lib <- Sys.getenv("R_LIBS_USER")
if (project_lib == "") {
  project_lib <- file.path(getwd(), "r_libs")
}
if (!dir.exists(project_lib)) dir.create(project_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(project_lib, .libPaths()))

safe_install_and_load <- function(pkgs, lib = project_lib) {
  repos <- "https://cloud.r-project.org"
  for (p in pkgs) {
    if (!requireNamespace(p, quietly = TRUE)) {
      message(sprintf("Installing '%s' into '%s' ...", p, lib))
      tryCatch(
        install.packages(p, lib = lib, repos = repos, dependencies = TRUE),
        error = function(e) {
          stop(sprintf("Failed to install package '%s': %s\nHint: check internet/proxy and write perms for '%s'.",
                       p, e$message, lib), call. = FALSE)
        }
      )
    }
    suppressPackageStartupMessages(library(p, character.only = TRUE))
  }
}

safe_install_and_load(c("sdcMicro", "readr"))

# --------------------------
# Config (tweak if needed)
# --------------------------
CSV_OUTDIR      <- "outputs_step3"  # set to NULL for no CSV outputs
RARE_LEVEL_PCT  <- 1                # flag categorical levels with < 1% frequency
SMALL_K_CUTOFFS <- c(1, 2, 3, 5)    # report share of classes with k <= these
L_DIVERSITY_MIN <- 2                # minimum distinct 'treat' values per class
TV_THRESHOLD    <- 0.30             # t-closeness via Total Variation distance

# --------------------------
# 1) Load data (robust path)
# --------------------------
cand <- c("data/aids_original_data.csv", "aids_original_data.csv", "../aids_original_data.csv")
csv_path <- cand[file.exists(cand)][1]
if (is.na(csv_path)) {
  url_csv <- "https://raw.githubusercontent.com/octopize/avatar-paper/main/datasets/AIDS/aids_original_data.csv"
  dir.create("data", showWarnings = FALSE)
  csv_path <- "data/aids_original_data.csv"
  download.file(url_csv, destfile = csv_path, mode = "wb")
}
df <- readr::read_delim(csv_path, delim = ";", show_col_types = FALSE)
cat("✅ Loaded:", csv_path, " | rows:", nrow(df), "cols:", ncol(df), "\n")

# --------------------------
# 2) Variable choices (align with your prior steps)
# --------------------------
key_vars      <- c("age", "gender", "race")               # quasi-identifiers (QI)
num_vars      <- c("wtkg", "karnof", "preanti", "days")   # numerical vars (for sdc context)
sensitive_var <- "treat"                                  # sensitive attribute

# Validate columns exist
stopifnot(all(key_vars %in% names(df)))
stopifnot(all(num_vars %in% names(df)))
stopifnot(sensitive_var %in% names(df))

cat("\n🔸 QI:", paste(key_vars, collapse = ", "),
    "\n🔸 Num:", paste(num_vars, collapse = ", "),
    "\n🔸 Sensitive:", sensitive_var, "\n", sep = "")

# --------------------------
# 3) SDC object & core risks
# --------------------------
sdc <- sdcMicro::createSdcObj(dat = df, keyVars = key_vars, numVars = num_vars, sensibleVar = sensitive_var)
rk  <- sdcMicro::get.sdcMicroObj(sdc, type = "risk")

global_risk   <- rk$global$risk
expected_reid <- rk$global$risk_ER
indiv_risk    <- as.data.frame(rk$individual)

cat(sprintf("\n🌐 Global risk: %.2f%%", 100 * global_risk))
cat(sprintf("\n👥 Expected re-identifications: %.0f / %d\n", expected_reid, nrow(df)))

# --------------------------
# 4) Identify parameters that show risk
#     4a) Small-k classes (k-anonymity) with current QIs
# --------------------------
fc <- sdcMicro::freqCalc(as.data.frame(df), keyVars = key_vars)
fk <- fc$fk                                     # class size for each record
class_map <- as.data.frame(df[, key_vars, drop = FALSE])
class_map$k <- fk

# Shares of small-k classes
k_shares <- sapply(SMALL_K_CUTOFFS, function(th) mean(class_map$k <= th, na.rm = TRUE))
names(k_shares) <- paste0("share_k_le_", SMALL_K_CUTOFFS)

cat("\n📉 Small-k prevalence with QIs {", paste(key_vars, collapse = ", "), "}:\n", sep = "")
print(round(k_shares, 4))

# Top risky (smallest-k) classes
agg <- aggregate(rep(1, nrow(class_map)), by = class_map[key_vars], FUN = length)
names(agg)[ncol(agg)] <- "k"
agg <- agg[order(agg$k, decreasing = FALSE), ]
head_risky <- head(agg, 20)
cat("\n🔎 Example high-risk (small-k) equivalence classes (top 20):\n")
print(head_risky)

# --------------------------
#     4b) Rare levels in categorical/low-cardinality QIs
# --------------------------
is_low_card_num <- function(x, thr = 10) is.numeric(x) && length(unique(x)) <= thr
cat_QIs <- key_vars[sapply(key_vars, function(v) !is.numeric(df[[v]]) || is_low_card_num(df[[v]]))]

rare_tbl <- lapply(cat_QIs, function(v) {
  tab <- sort(table(df[[v]], useNA = "no"), decreasing = TRUE)
  pct <- 100 * tab / sum(tab)
  data.frame(variable = v,
             level    = names(tab),
             count    = as.integer(tab),
             percent  = as.numeric(pct),
             rare_flag = pct < RARE_LEVEL_PCT,
             row.names = NULL)
})
rare_tbl <- if (length(rare_tbl)) do.call(rbind, rare_tbl) else NULL
if (!is.null(rare_tbl)) {
  cat("\n⚠️ Rare levels (< ", RARE_LEVEL_PCT, "%) among categorical/low-cardinality QIs:\n", sep = "")
  print(subset(rare_tbl, rare_flag == TRUE))
}

# --------------------------
#     4c) Outliers in numeric QIs (|z| > 3)
# --------------------------
cont_QIs <- key_vars[sapply(key_vars, function(v) is.numeric(df[[v]]) && !is_low_card_num(df[[v]]))]
outlier_tbl <- lapply(cont_QIs, function(v) {
  z <- scale(df[[v]])
  data.frame(variable     = v,
             n_outliers   = sum(abs(z) > 3, na.rm = TRUE),
             pct_outliers = 100 * mean(abs(z) > 3, na.rm = TRUE))
})
outlier_tbl <- if (length(outlier_tbl)) do.call(rbind, outlier_tbl) else NULL
if (!is.null(outlier_tbl)) {
  cat("\n📏 Numeric QIs — outlier counts (|z| > 3):\n")
  print(outlier_tbl)
}

# --------------------------
#     4d) l-diversity and t-closeness (Total Variation) for 'treat'
# --------------------------
glob_tab <- prop.table(table(df[[sensitive_var]]))

ldiv_df <- aggregate(df[[sensitive_var]],
                     by = df[key_vars],
                     FUN = function(x) {
                       p <- prop.table(table(x))
                       ldiv <- length(p)   # l-diversity = number of distinct sensitive values
                       # Total Variation distance vs global
                       all_lvls <- union(names(p), names(glob_tab))
                       p_al <- numeric(length(all_lvls)); names(p_al) <- all_lvls; p_al[names(p)] <- as.numeric(p)
                       g_al <- numeric(length(all_lvls)); names(g_al) <- all_lvls; g_al[names(glob_tab)] <- as.numeric(glob_tab)
                       tv <- 0.5 * sum(abs(p_al - g_al))
                       c(ldiv = ldiv, tv = tv)
                     })
ldiv_df$ldiv <- sapply(ldiv_df$x, function(z) z["ldiv"])
ldiv_df$tv   <- sapply(ldiv_df$x, function(z) z["tv"])
ldiv_df$x <- NULL

flag_ldiv <- subset(ldiv_df, ldiv < L_DIVERSITY_MIN)
flag_tv   <- subset(ldiv_df, tv > TV_THRESHOLD)

cat(sprintf("\n🧪 l-diversity check (need ≥ %d): %d classes below threshold\n", L_DIVERSITY_MIN, nrow(flag_ldiv)))
cat(sprintf("🧪 t-closeness (TV > %.2f): %d classes exceed threshold\n", TV_THRESHOLD, nrow(flag_tv)))

# --------------------------
#     4e) Sensitivity: which added field worsens small-k shares?
# --------------------------
alt_sets <- list(
  "age+gender+race" = c("age", "gender", "race"),
  "+homo"           = c("age", "gender", "race", "homo"),
  "+hemo"           = c("age", "gender", "race", "hemo"),
  "+drugs"          = c("age", "gender", "race", "drugs")
)
sens_rows <- lapply(names(alt_sets), function(nm) {
  ks <- alt_sets[[nm]]
  if (!all(ks %in% names(df))) return(NULL)
  fk_alt <- sdcMicro::freqCalc(as.data.frame(df), keyVars = ks)$fk
  c(model        = nm,
    share_k1     = mean(fk_alt == 1,  na.rm = TRUE),
    share_k_le_2 = mean(fk_alt <= 2,  na.rm = TRUE),
    share_k_le_3 = mean(fk_alt <= 3,  na.rm = TRUE),
    share_k_le_5 = mean(fk_alt <= 5,  na.rm = TRUE))
})
sens_tbl <- if (length(sens_rows)) do.call(rbind, sens_rows) else NULL
if (!is.null(sens_tbl)) {
  cat("\n🧲 Sensitivity — adding a single extra field to QIs increases small-k shares:\n")
  print(round(as.data.frame(sens_tbl), 4))
}

# --------------------------
# 5) Minimal plots (display only; no files)
# --------------------------
cat("\n📊 Displaying plots...\n")
boxplot(indiv_risk[, 1],
        main = "Distribution of Individual Re-identification Risk",
        ylab = "Individual risk", col = "lightblue")

hist(indiv_risk[, 1],
     breaks = 30, col = "skyblue",
     main = "Histogram of Individual Re-identification Risk",
     xlab = "Individual risk")

hist(class_map$k,
     breaks = seq(0.5, max(class_map$k, na.rm = TRUE) + 0.5, by = 1),
     main   = "Distribution of Equivalence Class Sizes (k)",
     xlab   = "Class size (k)", col = "orange")
cat("✅ Plots displayed.\n")

# --------------------------
# 6) Actionable, data-driven recommendations (printed)
# --------------------------
cat("\n🧭 Recommendations (tied to findings):\n")
cat("• AGE: generalize to 5-year bands (e.g., 15–19, 20–24, …) to lift small-k classes and reduce outliers.\n")
if (!is.null(rare_tbl) && any(rare_tbl$rare_flag)) {
  cat("• RARE LEVELS (< ", RARE_LEVEL_PCT, "%): merge into broader groups, or apply PRAM (≈10–20%) to stabilize marginals.\n", sep = "")
}
if (!is.null(outlier_tbl) && any(outlier_tbl$n_outliers > 0)) {
  cat("• NUMERIC OUTLIERS: use microaggregation (k=3–5) or robust winsorizing (cap at 1st/99th pct) on numeric QIs.\n")
}
if (nrow(flag_ldiv) > 0) {
  cat("• l-DIVERSITY: some QI groups have <", L_DIVERSITY_MIN, " distinct 'treat' values → reduce QI granularity (age bands, merge rare categories) until satisfied.\n", sep = "")
}
if (nrow(flag_tv) > 0) {
  cat("• t-CLOSENESS: some QI groups’ 'treat' mix differs a lot from global (TV >", TV_THRESHOLD, "). Consider further generalization or targeted PRAM.\n", sep = "")
}
if (!is.null(sens_tbl)) {
  cat("• QI SENSITIVITY: if adding {homo|hemo|drugs} spikes small-k shares, avoid them as QIs; if needed analytically, PRAM that field and/or locally suppress smallest-k classes.\n")
}
cat("• Always drop explicit IDs (e.g., pidnum) and constants (e.g., zprior). Recompute risks after each change.\n")

# --------------------------
# 7) (Optional) concise CSV outputs
# --------------------------
if (!is.null(CSV_OUTDIR)) {
  if (!dir.exists(CSV_OUTDIR)) dir.create(CSV_OUTDIR, showWarnings = FALSE)
  risk_summary <- data.frame(
    global_risk   = global_risk,
    expected_reid = expected_reid,
    share_k1      = mean(class_map$k == 1),
    share_k_le_2  = mean(class_map$k <= 2),
    share_k_le_3  = mean(class_map$k <= 3),
    share_k_le_5  = mean(class_map$k <= 5)
  )
  write.csv(risk_summary,               file.path(CSV_OUTDIR, "step3_risk_summary.csv"), row.names = FALSE)
  write.csv(head_risky,                 file.path(CSV_OUTDIR, "step3_small_k_classes_top20.csv"), row.names = FALSE)
  if (nrow(flag_ldiv) > 0) write.csv(flag_ldiv, file.path(CSV_OUTDIR, "step3_ldiversity_flags.csv"),    row.names = FALSE)
  if (nrow(flag_tv)  > 0) write.csv(flag_tv,  file.path(CSV_OUTDIR, "step3_tcloseness_flags.csv"),     row.names = FALSE)
  if (!is.null(rare_tbl)) write.csv(rare_tbl, file.path(CSV_OUTDIR, "step3_rare_levels_categorical_QIs.csv"), row.names = FALSE)
  if (!is.null(sens_tbl)) write.csv(as.data.frame(sens_tbl), file.path(CSV_OUTDIR, "step3_QI_sensitivity_kshares.csv"), row.names = FALSE)
  cat("\n🗂  CSVs written to:", CSV_OUTDIR, "\n")
}

cat("\n🔍 End of Step 3.\n")

