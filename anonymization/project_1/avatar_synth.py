#!/usr/bin/env python3
"""
avatar_synth.py

Simple numeric-only implementation of the Avatar synthetic data method,
inspired by Guillaudeux et al. (2023), "Patient-centric synthetic data
generation, no reason to risk re-identification in biomedical data analysis".

This version:
  - Works on numeric columns (continuous + 0/1 indicators).
  - Uses standardized Euclidean space for neighbour search.
  - For each row, generates one avatar as a random local mixture of its k neighbors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class AvatarSynthConfig:
    k_neighbors: int = 20
    random_state: Optional[int] = 42
    eps_dist: float = 1e-8  # to avoid division by zero in distances


class AvatarSynthesizer:
    """
    Avatar synthesizer for numeric data.

    Typical use:
        synth = AvatarSynthesizer(AvatarSynthConfig(k_neighbors=20))
        synth.fit(df, numeric_cols=[...], exclude_cols=['pidnum'])
        df_avatar = synth.transform()
    """

    def __init__(self, config: AvatarSynthConfig):
        self.config = config
        self._rng = np.random.default_rng(config.random_state)

        # Fitted attributes
        self.numeric_cols: List[str] = []
        self.exclude_cols: List[str] = []
        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None
        self._Z: Optional[np.ndarray] = None
        self._df_orig: Optional[pd.DataFrame] = None

    # ---------- Public API ----------

    def fit(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
        exclude_cols: Optional[List[str]] = None,
    ) -> "AvatarSynthesizer":
        """
        Fit the synthesizer on a dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Original dataset.
        numeric_cols : list of str, optional
            Columns to treat as numeric. If None, all numeric dtypes except exclude_cols.
        exclude_cols : list of str, optional
            Columns to keep as-is and *not* use in the numeric latent space (e.g., IDs).
        """
        self._df_orig = df.copy()
        self.exclude_cols = list(exclude_cols or [])

        if numeric_cols is None:
            # Use all numeric columns except excluded
            numeric_cols = [
                c for c in df.columns
                if c not in self.exclude_cols and pd.api.types.is_numeric_dtype(df[c])
            ]
        self.numeric_cols = numeric_cols

        # Extract numeric matrix
        X = df[self.numeric_cols].to_numpy(dtype=float)

        # ---- NEW: simple mean imputation for missing values ----
        if np.isnan(X).any():
            col_means = np.nanmean(X, axis=0)
            # If a column is entirely NaN, nanmean returns NaN -> set to 0.0
            col_means = np.nan_to_num(col_means, nan=0.0)
            inds = np.where(np.isnan(X))
            X[inds] = col_means[inds[1]]
        # --------------------------------------------------------

        # Standardize
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=0)
        sigma[sigma == 0] = 1.0  # avoid division by zero
        Z = (X - mu) / sigma

        self._mu = mu
        self._sigma = sigma
        self._Z = Z

        return self

    def transform(self) -> pd.DataFrame:
        """
        Generate the avatar (synthetic) dataset.

        Returns
        -------
        df_avatar : pd.DataFrame
            Synthetic dataset with same columns as original.
        """
        if self._df_orig is None or self._Z is None:
            raise RuntimeError("Must call .fit(...) before .transform().")

        Z_syn = self._generate_avatar_latent(self._Z)
        X_syn = self._unstnd(Z_syn)

        df_avatar = self._df_orig.copy()
        df_avatar[self.numeric_cols] = self._postprocess_numeric(X_syn)

        # At this stage, non-numeric / excluded columns are just copied over.
        # A full FAMD-based implementation would also synthesize them.
        return df_avatar

    # ---------- Internal helpers ----------

    def _unstnd(self, Z: np.ndarray) -> np.ndarray:
        """Undo standardization."""
        assert self._mu is not None and self._sigma is not None
        return Z * self._sigma + self._mu

    def _generate_avatar_latent(self, Z: np.ndarray) -> np.ndarray:
        """
        Core Avatar step: for each row, find k neighbors in Z-space and
        generate a random local mixture.
        """
        n, _ = Z.shape
        k = self.config.k_neighbors
        if k >= n:
            raise ValueError(f"k_neighbors={k} must be < number of rows n={n}")

        # Precompute squared norms for fast distance matrix
        # dist^2(i,j) = ||Z_i||^2 + ||Z_j||^2 - 2 Z_i·Z_j
        norms2 = np.sum(Z ** 2, axis=1)
        dist2 = norms2[:, None] + norms2[None, :] - 2 * (Z @ Z.T)
        dist2 = np.maximum(dist2, 0.0)
        dist = np.sqrt(dist2)
        np.fill_diagonal(dist, np.inf)  # ignore self-distance

        Z_syn = np.empty_like(Z)

        for i in range(n):
            d_i = dist[i]
            # indices of k nearest neighbors
            nn_idx = np.argpartition(d_i, k)[:k]
            d_nn = d_i[nn_idx]

            # distance weight
            D = 1.0 / (d_nn + self.config.eps_dist)

            # random weight (Exponential(1))
            R = self._rng.exponential(scale=1.0, size=k)

            # rank-based contribution: shuffle ranks 1..k, then use 1/2^rank
            ranks = self._rng.permutation(np.arange(1, k + 1))
            C = 1.0 / (2.0 ** ranks)

            P = D * R * C
            w = P / P.sum()

            Z_syn[i] = (w[:, None] * Z[nn_idx]).sum(axis=0)

        return Z_syn

    def _postprocess_numeric(self, X_syn: np.ndarray) -> np.ndarray:
        """
        Post-process numeric columns:
          - For binary columns (0/1 in original), threshold at 0.5.
          - For integer columns, round to nearest integer.
        """
        df = self._df_orig
        assert df is not None

        X_out = X_syn.copy()
        for j, col in enumerate(self.numeric_cols):
            orig_col = df[col]
            unique_vals = pd.unique(orig_col.dropna())
            # binary 0/1?
            if set(unique_vals).issubset({0, 1}):
                X_out[:, j] = (X_out[:, j] >= 0.5).astype(int)
            # integer-like but not just 0/1
            elif pd.api.types.is_integer_dtype(orig_col):
                X_out[:, j] = np.rint(X_out[:, j]).astype(int)
            # else: leave as float
        return X_out


# ---------- Simple evaluation helpers ----------

def compute_dcr_nndr(
    X_orig: np.ndarray,
    X_syn: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Distance to Closest Record (DCR) and Nearest Neighbor Distance Ratio (NNDR)
    between synthetic and original numeric matrices.

    Returns
    -------
    dcr_stats : pd.DataFrame
        Quantiles for DCR.
    nndr_stats : pd.DataFrame
        Quantiles for NNDR.
    """
    # Ensure float arrays
    X_orig = np.asarray(X_orig, dtype=float)
    X_syn = np.asarray(X_syn, dtype=float)

    # ---- NEW: mean-impute NaNs for distance computation ----
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
    # --------------------------------------------------------

    n_syn = X_syn.shape[0]
    n_orig = X_orig.shape[0]

    # Distances from each synthetic row to all original rows
    norms_syn = np.sum(X_syn ** 2, axis=1)
    norms_orig = np.sum(X_orig ** 2, axis=1)
    dist2 = norms_syn[:, None] + norms_orig[None, :] - 2 * (X_syn @ X_orig.T)
    dist2 = np.maximum(dist2, 0.0)
    dist = np.sqrt(dist2)

    # sort along original-axis
    sorted_d = np.sort(dist, axis=1)
    d1 = sorted_d[:, 0]
    d2 = sorted_d[:, 1]

    dcr = d1
    nndr = d1 / (d2 + 1e-8)

    def q_stats(x: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "median": [np.median(x)],
                "q05": [np.quantile(x, 0.05)],
                "q95": [np.quantile(x, 0.95)],
                "min": [x.min()],
                "max": [x.max()],
            }
        )

    return q_stats(dcr), q_stats(nndr)

