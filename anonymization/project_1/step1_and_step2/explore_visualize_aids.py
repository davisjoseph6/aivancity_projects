# Step 3 — Identify risk drivers + hypothesize anonymization (SAVE LOCALLY)
# Runs in RStudio. Produces a small, relevant set of PNGs + CSVs and prints results to console.

suppressPackageStartupMessages({
  if (!requireNamespace("sdcMicro", quietly = TRUE)) {
    install.packages("sdcMicro", repos = "https://cloud.r-project.org")
  }
  library(sdcMicro)
})

# ---------------------------
# Config (edit if you want)
# ---------------------------
LOCAL_PREF_PATH <- "C:/Users/davis/Downloads/aids_original_data.csv"  # try this first on Windows
OUTPUT_DIR      <- file.path("outputs")
PLOTS_DIR       <- file.path(OUTPUT_DIR, "plots")
CSV_DIR         <- file.path(OUTPUT_DIR, "csv")
URL_CSV         <- "https://raw.githubusercontent.com/octopize/avatar-paper/main/datasets/AIDS/aids_original_data.csv"
KEY_VARS        <- c("age", "gender", "race")                      # quasi-identifiers
NUM_VARS        <- c("wtkg", "karnof", "preanti", "days")          # numeric context
SENSITIVE_VAR   <- "treat"                                         # sensitive attribute
SMALL_K_THRESH  <- 5                                               # threshold for 'small' k
TOP_N_RISKY     <- 10                                              # how many top risky classes to list/save

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(PLOTS_DIR,  showWarnings = FALSE, recursive = TRUE)
dir.create(CSV_DIR,    showWarnings = FALSE, recursive = TRUE)

message("🔹 Loading dataset ...")
csv_path <- NULL
if (file.exists(LOCAL_PREF_PATH)) {
  csv_path <- LOCAL_PREF_PATH
} else if (file.exists("data/aids_original_data.csv")) {
  csv_path <- "data/aids_original_data.csv"
} else if (file.exists("aids_original_data.csv")) {
  csv_path <- "aids_original_data.csv"
} else {
  dir.create("data", showWarnings = FALSE)
  csv_path <- "data/aids_original_data.csv"
  download.file(URL_CSV, destfile = csv_path, mode = "wb", quiet = TRUE)
  message(paste0("⬇️  Downloaded to: ", normalizePath(csv_path)))
}

df <- read.delim(csv_path, sep = ";", dec = ".", stringsAsFactors = FALSE)
message(sprintf("✅ Loaded: %d rows × %d columns from %s", nrow(df), ncol(df), csv_path))
cat("\nPreview (first 5 rows):\n")
print(utils::head(df, 5))

# ---------------------------
# Build SDC object
# ---------------------------
sdc <- createSdcObj(
  dat         = df,
  keyVars     = KEY_VARS,
  numVars     = NUM_VARS,
  sensibleVar = SENSITIVE_VAR
)

# ---------------------------
# Risk measures
# ---------------------------
rk  <- get.sdcMicroObj(sdc, type = "risk")
global_risk   <- rk$global$risk
expected_reid <- rk$global$risk_ER
indiv_risk    <- as.data.frame(rk$individual)
colnames(indiv_risk)[1] <- "individual_risk"

cat(sprintf("\n🌐 Global risk: %.2f%%\n", 100*global_risk))
cat(sprintf("👥 Expected re-identifications: %.0f out of %d\n", expected_reid, nrow(df)))

# Equivalence classes over QIs
fc   <- freqCalc(df, keyVars = KEY_VARS)
kvec <- fc$fk
k_df <- cbind(df[, KEY_VARS, drop = FALSE], k = kvec)

pct_unique <- mean(kvec == 1, na.rm = TRUE)
pct_k_le_t <- mean(kvec <= SMALL_K_THRESH, na.rm = TRUE)
cat(sprintf("\n🔎 Percent unique on (%s): %.2f%%\n", paste(KEY_VARS, collapse=", "), 100*pct_unique))
cat(sprintf("🔎 Percent with k ≤ %d: %.2f%%\n", SMALL_K_THRESH, 100*pct_k_le_t))

# Top risky classes (unique combinations ranked by highest individual risk, then smallest k)
top_risk_tbl <- data.frame(df[, KEY_VARS, drop = FALSE],
                           k = kvec,
                           individual_risk = indiv_risk$individual_risk)
top_risk_tbl <- top_risk_tbl[order(-top_risk_tbl$individual_risk, top_risk_tbl$k), ]
top_risk_unique <- unique(top_risk_tbl)
cat("\nSmallest / riskiest equivalence classes (top 10):\n")
print(utils::head(top_risk_unique, TOP_N_RISKY), row.names = FALSE)

# Levels with very small counts per key var
cat("\n⚠️ Levels with very small counts (≤ ", SMALL_K_THRESH, ") for each key variable:\n", sep = "")
small_levels_list <- lapply(KEY_VARS, function(v) {
  tb <- sort(table(df[[v]]), decreasing = FALSE)
  small <- tb[tb <= SMALL_K_THRESH]
  data.frame(variable = v,
             level    = names(small),
             count    = as.integer(small),
             row.names = NULL)
})
names(small_levels_list) <- KEY_VARS
small_levels_df <- do.call(rbind, small_levels_list)
if (nrow(small_levels_df) == 0) {
  cat(" • None found at this threshold.\n")
} else {
  print(small_levels_df)
}

# Risk snapshots across different key sets
risk_over_keys <- function(keys) {
  fk <- freqCalc(df, keyVars = keys)$fk
  data.frame(
    keys        = paste(keys, collapse = "+"),
    pct_unique  = round(100*mean(fk == 1), 2),
    median_k    = stats::median(fk),
    pct_k_le_T  = round(100*mean(fk <= SMALL_K_THRESH), 2),
    row.names   = NULL
  )
}
key_tests <- list(
  c("age"), c("gender"), c("race"),
  c("age","gender"), c("age","race"), c("gender","race"),
  c("age","gender","race")
)
cat("\n📐 Risk snapshots for different key combinations:\n")
snap_df <- do.call(rbind, lapply(key_tests, risk_over_keys))
print(snap_df, row.names = FALSE)

# ---------------------------
# Inference/linkability check (ARMS → TREAT)
# ---------------------------
if (all(c("arms","treat") %in% names(df))) {
  cat("\n🧩 Inference check: does 'arms' reveal 'treat'?\n")
  xt <- table(arms = df$arms, treat = df$treat)
  print(xt)
  prop_treat1 <- apply(xt, 1, function(x) if (sum(x)==0) NA_real_ else x["1"]/sum(x))
  cat("P(treat=1 | arms):\n")
  print(round(prop_treat1, 4))
  if (all(prop_treat1 %in% c(0,1), na.rm = TRUE)) {
    cat("⚠️ Finding: TREAT is perfectly determined by ARMS → attribute disclosure risk.\n")
    cat("   Recommendation: do NOT release both; drop 'arms' or 'treat', or merge arms to break determinism.\n")
  }
}

# ---------------------------
# Helpers: Save CSV + Plot
# ---------------------------
save_csv <- function(df_, filename, index = TRUE) {
  path <- file.path(CSV_DIR, filename)
  # If row names are not useful, write without them
  if (index) {
    write.csv(df_, path, row.names = TRUE)
  } else {
    write.csv(df_, path, row.names = FALSE)
  }
  cat(paste0("💾 SAVED CSV: ", normalizePath(path), " (", nrow(df_), " rows, ", ncol(df_), " cols)\n"))
}

save_png_plot <- function(filename, plot_expr, caption = NULL, width = 1200, height = 900, res = 144) {
  path <- file.path(PLOTS_DIR, filename)
  png(path, width = width, height = height, res = res)
  op <- par(no.readonly = TRUE)
  on.exit({ par(op); dev.off() }, add = TRUE)
  # Add a little bottom space for caption
  par(mar = c(6.5, 5, 4, 2) + 0.1)
  eval.parent(substitute(plot_expr))
  if (!is.null(caption) && nchar(caption) > 0) {
    mtext(caption, side = 1, line = 4.5, cex = 0.9)
  }
  dev.off()
  cat(paste0("🖼️  SAVED PNG: ", normalizePath(path), "\n"))
}

# ---------------------------
# Save plots (with plain-language titles/captions)
# ---------------------------
# 1) Boxplot of individual risk
save_png_plot(
  "01_boxplot_individual_risk.png",
  {
    boxplot(indiv_risk$individual_risk,
            main = "How risky are individual records? (Boxplot)",
            ylab = "Re-identification risk per person",
            col  = "lightblue")
  },
  caption = "The box shows the 'typical' range of risks; dots beyond whiskers indicate unusually high/low risks."
)

# 2) Histogram of individual risk
save_png_plot(
  "02_hist_individual_risk.png",
  {
    hist(indiv_risk$individual_risk,
         breaks = 30,
         col    = "skyblue",
         main   = "How many people have each risk level? (Histogram)",
         xlab   = "Re-identification risk per person",
         ylab   = "Number of participants")
  },
  caption = "Bars show how many participants fall into each risk band. Taller bars = more people at that risk."
)

# 3) Histogram of equivalence class sizes (k)
save_png_plot(
  "03_hist_equivalence_class_sizes_k.png",
  {
    hist(kvec,
         breaks = seq(0.5, max(kvec, na.rm = TRUE)+0.5, by = 1),
         col    = "orange",
         main   = "How many people share the same profile? (k-anonymity)",
         xlab   = "k = number of people with the same (age, gender, race) pattern",
         ylab   = "Number of participants")
  },
  caption = paste0("Small k (e.g., k ≤ ", SMALL_K_THRESH,
                   ") means very specific profiles—higher disclosure risk.")
)

# ---------------------------
# Save CSVs (concise, relevant)
# ---------------------------
# 01: risk summary
risk_summary <- data.frame(
  global_risk_percent = round(100*global_risk, 2),
  expected_reidentifications = round(expected_reid, 0),
  percent_unique_on_keys     = round(100*pct_unique, 2),
  percent_k_le_threshold     = round(100*pct_k_le_t, 2),
  threshold_k                = SMALL_K_THRESH,
  row.names = NULL
)
save_csv(risk_summary, "01_risk_summary.csv", index = FALSE)
cat("\nRisk summary:\n"); print(risk_summary)

# 02: top risky classes
save_csv(utils::head(top_risk_unique, TOP_N_RISKY), "02_top10_risky_classes.csv", index = FALSE)

# 03: small count levels per key var
if (nrow(small_levels_df) == 0) {
  # write an empty frame with headers so users still get a file
  small_levels_df <- data.frame(variable = character(), level = character(), count = integer())
}
save_csv(small_levels_df, "03_small_count_levels_keyvars.csv", index = FALSE)

# 04: risk snapshots across key sets
save_csv(snap_df, "04_risk_snapshots_keysets.csv", index = FALSE)

# 05: inference crosstab (only if both exist)
if (all(c("arms","treat") %in% names(df))) {
  xt <- table(arms = df$arms, treat = df$treat)
  xt_df <- as.data.frame.matrix(xt)
  save_csv(xt_df, "05_crosstab_arms_treat.csv", index = TRUE)
}

# ---------------------------
# Method suggestions (plain language)
# ---------------------------
cat("\n🧭 Method suggestions (based on diagnostics):\n")
if (pct_unique > 0.01) {
  cat("• AGE: Many unique/small groups → band ages (e.g., 5-year bins) with globalRecode().\n")
}
cat("• RACE: If rare levels exist (count ≤ ", SMALL_K_THRESH, "), collapse to 'Other' or use PRAM to add uncertainty.\n", sep = "")
cat("• GENDER: Usually OK, but PRAM if tiny groups remain unique.\n")
cat("• NUMERICS (wtkg, preanti, cd4*): consider microaggregation (mdav), rounding (e.g., days to weeks), and top/bottom coding of extremes.\n")
if (all(c("arms","treat") %in% names(df))) {
  cat("• ARMS vs TREAT: Do not release both. Drop one or combine arms; otherwise TREAT is perfectly inferable.\n")
}
cat("• After coarsening, enforce k≥", SMALL_K_THRESH, " with localSuppression().\n", sep = "")

cat("\n✅ Done. Plots in ", normalizePath(PLOTS_DIR),
    " and CSVs in ", normalizePath(CSV_DIR), ".\n", sep = "")

